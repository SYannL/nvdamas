from .alfworld_prompt import alfworld_solver_system_prompt, alfworld_few_shots
from .fever_prompt import fever_solver_system_prompt, fever_few_shots
from .huskyqa_prompt import huskyqa_solver_system_prompt, huskyqa_few_shots
from .medmcqa_prompt import medmcqa_solver_system_prompt, medmcqa_few_shots
from .mtmind2web_prompt import mtmind2web_solver_system_prompt, mtmind2web_few_shots
from .pddl_prompt import pddl_prompts
from .scienceworld_prompt import scienceworld_solver_system_prompt, scienceworld_few_shots

def get_dataset_system_prompt(task: str, task_config: dict) -> str:
    prompt_map: dict = {
        'alfworld': alfworld_solver_system_prompt,
        'fever': fever_solver_system_prompt,
        'fever_aca_test': fever_solver_system_prompt,
        'fever_oral_test': fever_solver_system_prompt,
        'fever_ab_train_a': fever_solver_system_prompt,
        'fever_ab_train_b': fever_solver_system_prompt,
        'fever_ab_test_a': fever_solver_system_prompt,
        'fever_ab_test_b': fever_solver_system_prompt,
        'huskyqa': huskyqa_solver_system_prompt,
        'huskyqa_aca_test': huskyqa_solver_system_prompt,
        'pddl': pddl_prompts,
        'pddl_ab_train_A': pddl_prompts,
        'pddl_ab_train_B': pddl_prompts,
        'medmcqa_test': medmcqa_solver_system_prompt,
        'medmcqa_physio_30': medmcqa_solver_system_prompt,
        'medmcqa_pharma_30': medmcqa_solver_system_prompt,
        'medmcqa_physio_150_build': medmcqa_solver_system_prompt,
        'medmcqa_pharma_150_build': medmcqa_solver_system_prompt,
        'medmcqa_physio_20_test': medmcqa_solver_system_prompt,
        'medmcqa_pharma_20_test': medmcqa_solver_system_prompt,
        'medmcqa_anatomy_20': medmcqa_solver_system_prompt,
        'medmcqa_surgery_20': medmcqa_solver_system_prompt,
        'mtmind2web_train': mtmind2web_solver_system_prompt,
        'mtmind2web_test_task': mtmind2web_solver_system_prompt,
        'mtmind2web_test_website': mtmind2web_solver_system_prompt,
        'mtmind2web_test_subdomain': mtmind2web_solver_system_prompt,
        'scienceworld_train': scienceworld_solver_system_prompt,
        'scienceworld_dev': scienceworld_solver_system_prompt,
        'scienceworld_test': scienceworld_solver_system_prompt,
        'scienceworld_domain_a_train': scienceworld_solver_system_prompt,
        'scienceworld_domain_b_train': scienceworld_solver_system_prompt,
        'scienceworld_domain_a_test': scienceworld_solver_system_prompt,
        'scienceworld_domain_b_test': scienceworld_solver_system_prompt,
    }

    if prompt_map.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    if task not in ('pddl', 'pddl_ab_train_A', 'pddl_ab_train_B'):
        return prompt_map.get(task)
    else:
        task_type: str = task_config.get('game_name')
        return pddl_prompts[task_type]['instruction']
        


def get_task_few_shots(dataset: str, task_config: dict, few_shots_num: int) -> list[str]:
    
    if dataset == 'alfworld':
        task_type = task_config.get('task_type')
        if task_type is None:
            raise ValueError('The task config must have the `task_type` attribute.')
        return [alfworld_few_shots[f'react_{task_type}_2'], alfworld_few_shots[f'react_{task_type}_0']][:few_shots_num]
    
    elif dataset == 'fever':
        return fever_few_shots[:few_shots_num]

    elif dataset == 'fever_aca_test':
        return fever_few_shots[:few_shots_num]

    elif dataset == 'fever_oral_test':
        return fever_few_shots[:few_shots_num]
    
    elif dataset == 'fever_ab_train_a':
        return fever_few_shots[:few_shots_num]

    elif dataset == 'fever_ab_train_b':
        return fever_few_shots[:few_shots_num]

    elif dataset == 'fever_ab_test_a':
        return fever_few_shots[:few_shots_num]

    elif dataset == 'fever_ab_test_b':
        return fever_few_shots[:few_shots_num]
    
    elif dataset == 'huskyqa':
        return huskyqa_few_shots[:few_shots_num]

    elif dataset == 'huskyqa_aca_test':
        return huskyqa_few_shots[:few_shots_num]

    elif dataset in ('pddl', 'pddl_ab_train_A', 'pddl_ab_train_B'):
        task_type = task_config.get('game_name')
        if task_type is None:
            raise ValueError('The task config must have the `game_name` attribute.')
        return pddl_prompts[task_type]['examples'][:few_shots_num]
    
    elif dataset == 'medmcqa_test':
        return medmcqa_few_shots[:few_shots_num]
    
    elif dataset == 'medmcqa_physio_30':
        return medmcqa_few_shots[:few_shots_num]
    
    elif dataset == 'medmcqa_pharma_30':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_physio_150_build':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_pharma_150_build':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_physio_20_test':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_pharma_20_test':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_anatomy_20':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'medmcqa_surgery_20':
        return medmcqa_few_shots[:few_shots_num]

    elif dataset == 'mtmind2web_train':
        return mtmind2web_few_shots[:few_shots_num]

    elif dataset == 'mtmind2web_test_task':
        return mtmind2web_few_shots[:few_shots_num]

    elif dataset == 'mtmind2web_test_website':
        return mtmind2web_few_shots[:few_shots_num]

    elif dataset == 'mtmind2web_test_subdomain':
        return mtmind2web_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_train':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_dev':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_test':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_domain_a_train':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_domain_b_train':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_domain_a_test':
        return scienceworld_few_shots[:few_shots_num]

    elif dataset == 'scienceworld_domain_b_test':
        return scienceworld_few_shots[:few_shots_num]
    
    else:
        raise ValueError(f'Unsupported dataset type: {dataset}')

