# tocolo

## dataset

The data required for reasoning to fine-tune and our statistics on the dataset are available at the link: https://pan.baidu.com/s/18lBBonFO4fF0aGkBXv15Qw. 

Extract password: vksw.

### description file for code file:
k9mail_filename_intro_for_chatgbt.json

AntennaPod_filename_intro_for_chatgbt.json

cgeo_filename_intro_for_chatgbt.json

anki_filename_intro_for_chatgbt.json

termux_filename_intro_for_chatgbt.json

## fine-tune
--project_name=anki

bash train.sh

## inference
--project_name=anki

bash test.sh
