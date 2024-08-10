import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog="eval", description="Evaluate about the inference code.")
print('eval.py started')
def evaluate_result():
    # Load the correct answer from correct_answer.json
    with open('/root/Korean_CCI_2024_Wittgenstein/result_cat_v3_ckpt_17.json', 'r') as file:
        correct_answers = json.load(file)

    # Load the inference result from the current path
    with open('/root/Korean_CCI_2024_Wittgenstein/resource/data/correct_answer.json', 'r') as file:
        results = json.load(file)

    # Compare the result with the correct answer
    answer_list = []
    ground_truth_list = []
    category_list = []
    result_list = []
    for i, result in enumerate(correct_answers):
        # breakpoint()
        answer_output = results[i]['output']
        ground_truth = result['output']
        category = result['input']['category']
        answer_list.append(results[i]['output'])
        ground_truth_list.append(result['output'])
        category_list.append(result['input']['category'])
        result_list.append((answer_output, ground_truth, category))

        print(f'Answer: {answer_list[-1]}, Ground Truth: {ground_truth_list[-1]}, Correct: {"O" if answer_list[-1] == ground_truth_list[-1] else "x"}')
    
    correct_category = {
        "반응":0,
        "원인":0,
        "동기":0,
        "후행사건":0,
        "전제":0
    }
    incorrect_category = {
        "반응":0,
        "원인":0,
        "동기":0,
        "후행사건":0,
        "전제":0
    }
    
    # Count the number of correct answers
    correct_count = 0
    incorrect_count = 0
    for result in result_list:
        category = result[2]
        if result[0] == result[1]:
            correct_category[category] += 1
            correct_count += 1
        else:
            incorrect_category[category] += 1
            incorrect_count += 1

    print(f"반응 : {correct_category['반응'] / (incorrect_category['반응']+correct_category['반응']):.2f}%")
    print(f"원인 : {correct_category['원인'] / (incorrect_category['원인']+correct_category['원인']):.2f}%")
    print(f"동기 : {correct_category['동기'] / (incorrect_category['동기']+correct_category['동기']):.2f}%")
    print(f"후행사건 : {correct_category['후행사건'] / (incorrect_category['후행사건']+correct_category['후행사건']):.2f}%")
    print(f"전제 : {correct_category['전제'] / (incorrect_category['전제']+correct_category['전제']):.2f}%")

    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {correct_count / (correct_count + incorrect_count):.2f}%")
    # # Plot the result
    # plt.plot(result, label='Result')
    # plt.plot(correct_answer, label='Correct Answer')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.legend()

    # # Save the plot as png
    # plt.savefig('result_plot.png')

    # # Show the plot
    # plt.close()

# Call the evaluate_result function
evaluate_result()
print('eval.py finished')