import json
from itertools import islice

# SUBREDDIT = 'The_Donald' # Already done: 30
ALREADY_DONE = 30 # How many iterations are already processed?
SUBREDDIT = 'TwoXChromosomes'
ALREADY_DONE = 300

OMIT_REMOVED = True
FIELDS_SAVED = {'body', 'controversiality', 'score', 'subreddit'}

def extract_json(overwrite=False):
    if overwrite:
        open('generator/datasets/' + SUBREDDIT + '.json', 'w').close()

    with open('generator/datasets/RC_2017-06') as f:
        n = 10000
        
        # Iterate over n lines at a time for memory's sake
        for i in range(ALREADY_DONE, ALREADY_DONE + 50):
            lines_gen = islice(f, i*n, (i+1)*n)
            new_lines = ''
            for line in lines_gen:
                j_content = json.loads(line)
                if j_content['subreddit'] == SUBREDDIT:
                    if OMIT_REMOVED and (j_content['body'] == '[removed]' or j_content['body'] == '[deleted]'):
                        continue
                    new_line = json.dumps({key: val for (key, val) in j_content.items() if key in FIELDS_SAVED})
                    new_lines += new_line + '\n'
            # Save in every iteration for robustness
            with open('datasets/' + SUBREDDIT + '.json', 'a') as f_out:
                f_out.write(new_lines)
            print('Finished iteration ' + str(i))

def json_to_corpus():
    with open('generator/datasets/' + SUBREDDIT + '.json', 'r') as f:
        lines = f.readlines()
        with open('generator/datasets/' + SUBREDDIT + '.txt', 'w+', encoding='utf-8') as f:
            for line in lines:
                line_data = json.loads(line)
                line_text = line_data['body'].rstrip()
                line_text = line_text.replace('\n', ' ')
                f.write(line_text + '\n')

# Execute here
# extract_json(overwrite=False)
# json_to_corpus()
# print('Done')
