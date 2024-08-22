# python src/ensemble_eval.py --files /root/Korean_CCI_2024_Wittgenstein/result_cat_v5_ckpt_35.json /root/Korean_CCI_2024_Wittgenstein/result_cat_v5_ckpt_29.json /root/Korean_CCI_2024_Wittgenstein/result_cat_v5_ckpt_23.json /root/Korean_CCI_2024_Wittgenstein/result_cat_v5_ckpt_17.json /root/Korean_CCI_2024_Wittgenstein/result_cat_v5_ckpt_41.json

import json
import argparse
from collections import Counter, defaultdict
from itertools import combinations

parser = argparse.ArgumentParser(prog="eval", description="Evaluate about the inference code with ensemble.")
parser.add_argument('--files', nargs='+', required=True, help="List of JSON result files for ensemble")
print('ensemble_eval.py started')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def evaluate_result(ensemble_results, correct_answers):
    correct_category = defaultdict(int)
    incorrect_category = defaultdict(int)

    correct_count = 0
    incorrect_count = 0
    incorrect_results = []

    for i, result in enumerate(correct_answers):
        answer_output = ensemble_results[i]['output']
        ground_truth = result['output']
        category = result['input']['category']

        if answer_output == ground_truth:
            correct_category[category] += 1
            correct_count += 1
        else:
            incorrect_category[category] += 1
            incorrect_count += 1
            result_with_model_output = result.copy()
            result_with_model_output['model_output'] = answer_output
            incorrect_results.append(result_with_model_output)

        print(f'[{i + 1}] Answer: {answer_output}, Ground Truth: {ground_truth}, Correct: {"O" if answer_output == ground_truth else "x"}')

    # Save incorrect results to a separate JSON file
    if incorrect_results:
        with open('/root/incorrect_results_ensemble.json', 'w') as outfile:
            json.dump(incorrect_results, outfile, ensure_ascii=False, indent=4)

    # Print category-wise results
    for category in correct_category.keys():
        total = correct_category[category] + incorrect_category[category]
        print(f"{category} : {100 * (correct_category[category] / total):.2f}%")

    total_count = correct_count + incorrect_count
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {100 * (correct_count / total_count):.2f}%")

    return correct_count, total_count

def ensemble_predictions(json_files):
    ensemble_results = []
    num_samples = len(json_files[0])

    for i in range(num_samples):
        outputs = [json_files[j][i]['output'] for j in range(len(json_files))]
        most_common_output = Counter(outputs).most_common(1)[0][0]
        ensemble_results.append({'output': most_common_output})

    return ensemble_results

def main(args):
    json_files = [load_json(file_path) for file_path in args.files]
    correct_answers = load_json('/root/Korean_CCI_2024_Wittgenstein/resource/data/correct_answer.json')

    best_correct = 0
    best_combination = None

    # Iterate over all possible combinations of the files
    for r in range(3, len(json_files) + 1):
        for combination in combinations(json_files, r):
            ensemble_results = ensemble_predictions(combination)
            correct_count, total_count = evaluate_result(ensemble_results, correct_answers)

            if correct_count > best_correct:
                best_correct = correct_count
                best_combination = combination

    if best_combination:
        print(f"Best ensemble combination was from {len(best_combination)} files with {best_correct} correct out of {total_count}.")
        print(f"Files used in the best combination: {[args.files[json_files.index(file)] for file in best_combination]}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print('ensemble_eval.py finished')
