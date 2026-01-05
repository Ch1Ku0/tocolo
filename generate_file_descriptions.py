import os
import openai
import json



openai.api_key = "your_key"



openai.base_url = "openai.base_url"
openai.default_headers = {"x-foo": "true"}


def build_file_introdunction(project_name):


    version_1_prompt = '''You are analyzing a source code file from the app ''' + project_name + '''
    The following is the file path and the list of methods it contains:

    File Path: ''' '''
    Methods and parameters: ''' '''

    Based on the provided file path and the list of methods with their parameters, summarize the functionality of this file in a description. The entire output must be one single line.
    '''

    version_2_prompt = '''You are analyzing a source code file from the app ''' + project_name + '''
    The following is the file path and the list of methods it contains:

    File Path: ''' '''
    Methods and parameters: ''' '''

    Based on the provided file path and the list of methods with their parameters, generate a concise description of what this file does, including its main functionality, its responsibility within the project, and the scenarios in which developers may need to modify this file. The entire output must be one single line    
    '''


    version_3_prompt = '''You are analyzing a source code file from the app ''' + project_name + '''
    The following is the file path and the list of methods it contains:

    File Path: ''' '''
    Methods and parameters: ''' '''

    Based on the provided file path and the list of methods with their parameters, generate a description of this file from three perspectives:
    Function: Describe the main functionality of this file using the provided methods and their parameters. Focus on what the file actually does, how the methods contribute to its core behavior, and what responsibilities the file carries in terms of computation, data handling, UI interaction, or other logic.
    Role: Explain the file's responsibility or position within the app's architecture or module structure. Consider how this file interacts with other components, whether it serves as a controller, helper, service, or interface layer, and its importance in the overall app workflow. Use clear and precise language, and avoid general statements. 
    Modification Scenarios: Describe typical situations in which developers would modify this file. Include updates for adding new features, fixing bugs, improving performance, adapting to API changes, handling new input/output requirements, or addressing security concerns. Focus on realistic developer actions directly related to the provided methods.
    Strict requirements:
    - Output must follow exactly this format:
    Function: [description].  Role: [description].  Modification Scenarios: [description].
    - The entire output must be one single line.
    - Do not invent method names, variables, classes, or implementation details not included in the method list.
    - Do not use quotation marks, bullet points, or line breaks.
    - Base descriptions only on the provided method names, file path, and project summary.
    '''
    
    user_query = "You are an AI mobile app project analysis assistant. Please analyze the file structure tree of the app " + project_name + "identify its modules and functions, and provide a comprehensive understanding of the project. The following is the project’s structure tree: " + app_tree_dict[project_name] 


    app_tree_dict = {
        "project_name": "",
    }

    completion = openai.chat.completions.create(
        model="gpt-5-2025-08-07",
        max_completion_tokens=512,
        temperature=0,
        presence_penalty = 0,
        frequency_penalty = 0,
        top_p = 1,
        messages=[{
                    "role": "user",
                    "content": user_query,
                },
        ],
    )

    project_assistant_answer  = completion.choices[0].message.content


    completion = openai.chat.completions.create(
        model="gpt-5-2025-08-07",
        max_completion_tokens=512,
        temperature=0,
        presence_penalty = 0,
        frequency_penalty = 0,
        top_p = 1,
        messages=[{"role":"user","content":user_query}, {"role":"assistant","content":project_assistant_answer},
            {
                "role": "user",
                "content": version_1_prompt,
            },
        ],
    )
    result_message = completion.choices[0].message.content
    print(result_message)



for project_name in ["project_name"]:
    build_file_introdunction(project_name)
