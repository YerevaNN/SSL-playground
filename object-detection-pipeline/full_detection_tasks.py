import os
import requests
import traceback

TEAM_SECRET = os.environ['LWLL_TA1_TEAM_SECRET']
GOV_SECRET = os.environ['LWLL_TA1_GOVTEAM_SECRET']
headers = {'user_secret': TEAM_SECRET, "govteam_secret":GOV_SECRET}
url = os.environ['LWLL_TA1_API_ENDPOINT']
r = requests.get(f"{url}/list_tasks", headers=headers)
tasks = r.json()['tasks']
image_classification_task = []

for task in tasks:
    r = requests.get(f"{url}/task_metadata/{task}", headers=headers)
    problem_type = r.json()['task_metadata']['problem_type']
    if problem_type == 'object_detection':
        image_classification_task.append(task)

for task in image_classification_task:
    try:
        print('================ Starting Task: {} ================'.format(task))
        os.system('python run_pipeline.py --task={} --session=coral-obj-det-{}'.format(task, task[:5]))
        print('================ Task: {} Has Finished ================'.format(task))
        print()
    except Exception as err:
        traceback.print_tb(err.__traceback__)
        print('================ Task: {} Failed. Skip to the next Task ================'.format(task))
        print()
