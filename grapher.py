import unlearn as unlearner
from itertools import product
import json
#unlearner.unlearn()
unlearner.unlearn(3,0,8e-3)

exit(0)


def generate_grad_map_permutations(base_grad_map=[8, 32, 2000], steps=[2, 8, 200], variations=3):
    # Generate ranges for each value based on step and variations
    ranges = [
        [base + i * step for i in range(-variations, variations + 1)]
        for base, step in zip(base_grad_map, steps)
    ]
    # Generate all permutations using product
    all_permutations = list(product(*ranges))
    return all_permutations

# Generate all permutations of target values
#grad_map_permutations=generate_grad_map_permutations()
#grad_maps= [[8,32,2000],[8,32,4000],[8,32,1000],[8,16,2000],[8,16,4000],[8,16,1000],[12,32,2000]]
grad_maps=[[8,32,2000]]
#all_targets = [(x, y) for x, y in product(range(10), range(10)) if x != y]
all_targets = [x for x in range(10) if x !=3]
learning_rates= [4e-3,8e-3,1e-2,2e-2,4e-2]
batch_sizes = [16]
epochs_forgets = [4]
epochs_relearns = [10]


# Result storage
results = []

# Iterate over all combinations
for target, lr, batch_size, epochs_forget, epochs_relearn,grad_map in product( all_targets, learning_rates, batch_sizes, epochs_forgets, epochs_relearns, grad_maps):

    #forget_target, sub_target = target
    sub_target=3
    # Call the unlearn function
    dataset_score,forgotten_score,new_data_score,static_score,starting_accuracy_forgotten = unlearner.unlearn(target,sub_target,lr,batch_size,epochs_forget,epochs_relearn,grad_map)

    # Store the parameters and outputs together
    results.append({
        "parameters": {
            "forget_target": target,
            "sub_target": sub_target,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs_forget": epochs_forget,
            "epochs_relearn": epochs_relearn,
            "grad_map2": grad_map
        },
        "results": {
            "dataset_score": dataset_score,
            "forgotten_score": forgotten_score,
            "new_data_score": new_data_score,
            "static_score": static_score,
            "starting_accuracy_forgotten": starting_accuracy_forgotten
        }
    })


# Save to a JSON file
with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Ablation study completed and saved to 'ablation_results.json'")
print(results)