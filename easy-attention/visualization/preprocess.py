import json

lines = open("training_log_evals.jsonl").readlines()

values = []
for l in lines:
    j = json.loads(l)
    values.append(
        {
            "num_examples_trained_on": j["num_examples_trained_on"],
            "latest_training_loss": j["latest_training_loss"],
            "eval_examples": list(
                map(
                    lambda ee: {
                        "eval_line": ee["eval_line"],
                        "solution_softmax": ee["predictions"][ee["solution"]],
                    },
                    j["eval_examples"],
                )
            ),
        }
    )

print(json.dumps(values))
