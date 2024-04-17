import matplotlib.pyplot as plt


def plot_losses(train_losses: list, test_losses: list):
    """
    Plot the training and testing losses over generations.

    Args:
    train_losses (list of floats): A list containing the loss values of the training set.
    test_losses (list of floats): A list containing the loss values of the testing set.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.title('Evolution of Train and Test Loss')
    plt.legend()
    plt.show()


def summarize_best_loss_performance(test_losses: list, time_list: list):
    """
    Print the best test loss performance details, including the iteration indexes and total time taken up to each
    iteration.

    Args:
    test_losses (list of floats): A list of test loss values across different generations.
    time_list (list of floats): A list of time durations for each generation to complete.
    """
    best_test_loss = min(test_losses)
    # Identify all indexes where the test loss is equal to the best test loss
    best_indexes = [i for i, loss in enumerate(test_losses) if loss == best_test_loss]
    # Calculate the cumulative time up to each best index
    total_times_up_to_best = [sum(time_list[:i + 1]) for i in best_indexes]

    print("Best Test Loss:", best_test_loss)
    print("Indexes of Best Test Loss:", best_indexes)
    print("Total Times up to these iterations (seconds):", total_times_up_to_best)
