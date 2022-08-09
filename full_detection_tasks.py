import os
import requests
import traceback

TEAM_SECRET = os.environ['LWLL_TA1_TEAM_SECRET']
GOV_SECRET = os.environ['LWLL_TA1_GOVTEAM_SECRET']
headers = {'user_secret': TEAM_SECRET, "govteam_secret":GOV_SECRET}
url = os.environ['LWLL_TA1_API_ENDPOINT']
task_id = os.environ['LWLL_TA1_PROB_TASK']
r = requests.get(f"{url}/list_tasks", headers=headers)
tasks = r.json()['tasks']
od_tasks = []

for task in tasks:
    r = requests.get(f"{url}/task_metadata/{task}", headers=headers)
    r = r.json()['task_metadata']
    problem_type = r['problem_type']
    if problem_type == 'object_detection':
        od_tasks.append(task)
        print(",".join([
            r['task_id'],
            r['base_dataset'],
            '/'.join([str(x) for x in r['base_label_budget_full']]),
            r['adaptation_dataset'],
            '/'.join([str(x) for x in r['adaptation_label_budget_full']]),
            ' / '.join(r['whitelist'])
        ]))

for task in od_tasks:
    if task_id.lower() != 'all' and task != task_id:
        continue

    try:
        print('================ Starting Task: {} ================'.format(task))
        cmd = 'python run_pipeline.py --task={} --session=coral-obj-det-{}'.format(task, task[:5])
#        os.system(cmd)
        print(cmd)
        print('================ Task: {} Has Finished ================'.format(task))
        print()
    except Exception as err:
        traceback.print_tb(err.__traceback__)
        print('================ Task: {} Failed. Skip to the next Task ================'.format(task))
        print()
