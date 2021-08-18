""" Pipeline Loop for ISI.

Reference: https://gitlab.lollllz.com/lwll/jpl_ta1 
"""

import glob
import os
import requests
import logging

from absl import app
from absl import flags
from tqdm import tqdm

#import active_strategies
#import models



URL_LOOKUP = {
  'dev': 'https://api-dev.lollllz.com',
  'staging': 'https://api-staging.lollllz.com',
  'prod': 'https://api-prod.lollllz.com',
}

TEAM_SECRET = os.environ['LWLL_TA1_TEAM_SECRET']
GOV_SECRET = os.environ['LWLL_TA1_GOVTEAM_SECRET']
LWLL_ENDPOINT = os.environ['LWLL_TA1_API_ENDPOINT']
#flags.DEFINE_string('task', 'problem_test_image_classification', '')
# flags.DEFINE_string('task', 'd48f8a99-ba12-4df8-a74a-d06413b0f1ba', '')
# flags.DEFINE_string('task', '4d924004-347e-4043-8dd2-f4f8b0f52efd', '')
flags.DEFINE_string('task', 'problem_test_obj_detection', '')
flags.DEFINE_string('ckpt_path', 'N/A', '')
flags.DEFINE_string('datasets_dir', '~/lwll/datasets/development',
                    'Directory where datasets are downloaded onto')
flags.DEFINE_string('data_type', 'full', 'One of "sample" or "full"')
flags.DEFINE_string('session', 'hcxie_detection', 'Name of session. Can be any string') 
#flags.DEFINE_string('model', 'RandomModel', 'Name of class in model.py')
#flags.DEFINE_string('active_strategy', 'RandomStrategy', 'Name class in active_strategies.py')

#flags.DEFINE_integer('num_epochs', 5, 'Number of train epochs')
#flags.DEFINE_integer('num_labels_per_epochs', 50,
#                     'Number of files to get labels for, per epoch')

flags.DEFINE_string('secret', '789e1a6c-e9bb-4135-b2d4-6ac20a38808c' ,
                    'The secret for the API')
flags.DEFINE_string('api', 'dev',
                    'One of "dev", "staging", "prod".')
flags.DEFINE_string('session_id', None, 'session id for continuing an existing session. Keep none if starting a new session')
flags.DEFINE_integer('full_batch_size', 200, 'Full Batch Size')
flags.DEFINE_integer('label_batch_size', 64, 'labeled batch size')
flags.DEFINE_string('alg', 'A', 'Which algorithm to use')
flags.DEFINE_string('read_outputs_from', None, 'If set, will read the outputs instead of training new models.')
flags.DEFINE_integer('skip_base_upto', None, 'If set, will skip base stages up to (including) specified stage.')

# Can add more flags (e.g. objective function).
FLAGS = flags.FLAGS

def post_json(uri, body_json, **additional_headers):
  headers = {'user_secret': TEAM_SECRET, "govteam_secret":GOV_SECRET}
  headers.update(additional_headers)
  url = LWLL_ENDPOINT
  r = requests.post(os.path.join(url, uri), json=body_json, headers=headers)
  return r.json()


def get_json(uri, **additional_headers):
  headers = {'user_secret': TEAM_SECRET, "govteam_secret":GOV_SECRET}
  headers.update(additional_headers)
  url = LWLL_ENDPOINT
  r = requests.get(os.path.join(url, uri), headers=headers)
  return r.json()

def create_default_task():
  tasks = get_json('list_tasks')['tasks']
  resume_session = False
  if FLAGS.task not in tasks:
    raise ValueError('Invalid task %s. Available tasks are: %s' % (
        FLAGS.task, ', '.join(tasks)))

  ### TEST.
  # Start a new session.
  if FLAGS.session_id == None:
    print('Start a new session.')
    session_token = post_json('auth/create_session', body_json={
      'session_name': FLAGS.session,
      'data_type': FLAGS.data_type,
      'task_id': FLAGS.task,
    })
    session_token = session_token['session_token']
  else:
    print('Resuming Session: ' + FLAGS.session_id)
    session_token = FLAGS.session_id
    resume_session = True

  print('Session Status: ')
  session_status = get_json('session_status', session_token=session_token)
  print(str(session_status))

  # Get the dataset and the directory. 
  current_dataset = session_status['Session_Status']['current_dataset']
  dataset_dir = os.path.join(
      os.path.expanduser(FLAGS.datasets_dir),
      current_dataset['uid'],
      '%s_%s' % (current_dataset['uid'], FLAGS.data_type))

  return FLAGS, resume_session, {
    'train_dir': os.path.join(dataset_dir, 'train'),
    'test_dir': os.path.join(dataset_dir, 'test'),
    'session': session_status,
    'budgets': session_status['Session_Status']['current_label_budget_stages'],
    'session_token': session_token,
    'num_classes': len(current_dataset['classes']),
  }


def run_test(stage_name, test_files, transform_fn, stack_fn, model_fn, batch_size=100):
  tt = tqdm(range(0, len(test_files), batch_size))
  all_x = []
  all_y = []
  for start_i in tt:
    end_i = min(start_i + batch_size, len(test_files))
    batch_x = list(map(transform_fn, test_files[start_i:end_i]))
    batch_x = stack_fn(batch_x)
    predictions = model_fn(batch_x)
    assert len(predictions) == len(test_files[start_i:end_i]), 'model function return list of size %i' % len(test_files[start_i:end_i])

    all_x += test_files[start_i:end_i]
    all_y += list([str(int(p)) for p in predictions])
    tt.set_description('Testing ' + stage_name)
  return {'class': all_y, 'id': list(map(os.path.basename, all_x))}


def run_loop(
    task,
    base_unsupervised_learn=None,
    base_supervised_learn=None,

    adapt_unsupervised_learn=None,
    adapt_supervised_learn=None,

    dataset=None,
    #make_dataset_fn=None.
    ):
  test_files = glob.glob(os.path.join(task['test_dir'], '*'))

  print('\n\nDoing unsupervised learning...')
  base_unsupervised_learn()   # Before requesting seed labels.

  def request_labels_fn(num_labels, active_learner):
    if task['session']['Session_Status']['budget_left_until_checkpoint'] == 0:
      print('ERROR: No more label budget for this round!')
      return
    
    num_labels = min(num_labels, task['session']['Session_Status']['budget_left_until_checkpoint'])
    selected_examples = active_learner.top_unlabeled(dataset, num_labels)

    label_response = post_json('query_labels', {'example_ids': selected_examples},
                               session_token=task['session_token'])
    dataset.register_labels(label_response['Labels'])
    # Refresh status and return budget remaining in this stage.
    task['session'] = get_json('session_status', session_token=task['session_token'])
    return task['session']['Session_Status']['budget_left_until_checkpoint']


  for stage, budget in enumerate(task['budgets']):
    if stage == 0:
      # Request seed labels.
      label_response = get_json('seed_labels', session_token=task['session_token'])
      dataset.register_labels(label_response['Labels'])
      budget = 0
      # Seed labels exhausted.
      task['session'] = get_json('session_status', session_token=task['session_token'])
      assert 0 == task['session']['Session_Status']['budget_left_until_checkpoint']
    elif stage == 1:
      # Request secondary seed labels.
      label_response = get_json('secondary_seed_labels', session_token=task['session_token'])
      dataset.register_labels(label_response['Labels'])
      budget = 0
      task['session'] = get_json('session_status', session_token=task['session_token'])
      assert 0 == task['session']['Session_Status']['budget_left_until_checkpoint']
    
    # Call the (semi-)supervised training and get classification function.
    print('\n\nInvoking base_supervised_learn for stage %i' % stage)
    classification_fn = base_supervised_learn(stage, budget, request_labels_fn)
    if classification_fn is None:
      raise ValueError('base_seed() must return a function that returns predicted classes.')
  
    # Get test predictions.
    predictions = run_test(
          'base_stage_%i' % stage,
          test_files,   #['f1.png', 'f2.png', ..]
          dataset.transforms['test_x'],  # [Tensor(H, W, C), ... ]
          dataset.stack_batch_fn,  # Tensor(B, H, W, C)
          classification_fn)   # # Tensor(B, H, W, C)  -->  Int Tensor(B)
    # Submit test predictions.
    task['session'] = post_json(
        'submit_predictions', {'predictions':predictions},
        session_token=task['session_token'])


  import IPython; IPython.embed()

  """
  ### (supervised) Training loop.
  current_labels = {}  # Filename to label.
  for epoch in range(FLAGS.num_epochs):
    logging.info('***** Start Epoch %i', epoch)

    # Get training samples, according to strategy.
    ranked_files = active_strategy.rank_files()
    new_ranked_files = []
    i = -1
    while len(new_ranked_files) < FLAGS.num_labels_per_epochs:
      i += 1
      if i >= len(ranked_files): break  # labeled all files!
      basename = os.path.basename(ranked_files[i])
      if basename in current_labels:
        continue

      new_ranked_files.append((basename, ranked_files[i]))

    import IPython; IPython.embed()
    label_response = (
        post_json(
          'query_labels', {'example_ids': [f[0] for f in new_ranked_files]},
          session_token=session_token)
    )

    print('Got %i labels' % len(label_response['Labels']))
    current_labels.update({example: label for label, example in label_response['Labels']})
    logging.info('@epoch:%i. Budget Used=%i', epoch, label_response['Session_Status']['budget_used'])

    # Train the model.
    labeled_train_files = []
    labels = []
    unlabeled_train_files = []
    for f in x_train_files:
      if os.path.basename(f) in current_labels:
        labeled_train_files.append(f)
        labels.append(current_labels[os.path.basename(f)])
      else:
        unlabeled_train_files.append(f)

    model.train(unlabeled_train_files, labeled_train_files, labels)
  
    predictions = model.test(x_test_files)
    prediction_dict = {'predictions': {'class': predictions, 'id': list(map(os.path.basename, x_test_files))}}

    ## TODO: Call only once budget is exhausted
    response = post_json('submit_predictions', prediction_dict, session_token=session_token)
    print('response: %s' % response)

  """



if __name__ == '__main__':
  app.run(main)

