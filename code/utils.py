import matplotlib.pyplot as plt


def plot_losses(train_losses: list, test_losses: list):
    """
    Plot the training and testing losses over generations.

    Args:
    train_losses (list of floats): A list containing the loss values of the training set.
    test_losses (list of floats): A list containing the loss values of the testing set.
    """
    plt.figure(figsize=(10, 5))
    generations = range(1, len(train_losses) + 1)
    plt.plot(generations, train_losses, label='Функція втрат для навчальної вибірки')
    plt.plot(generations, test_losses, label='Функція втрат для тестової вибірки')
    plt.xlabel('Ітерації')
    plt.ylabel('Значення функції втрат')
    plt.title('Зміна функції втрат для навчальної та тестової вибірок')
    plt.legend()
    plt.xticks(list(range(1, 2)) + list(range(5, len(train_losses) + 1, 5)))
    plt.show()


def summarize_best_loss_performance(test_losses: list, train_losses: list, time_list: list):
    """
    Print the best test loss performance details, including the iteration indexes and total time taken up to each
    iteration.

    Args:
    test_losses (list of floats): A list of test loss values across different generations.
    time_list (list of floats): A list of time durations for each generation to complete.
    """
    best_test_loss = min(test_losses)
    # Identify all indexes where the test loss is equal to the best test loss
    best_test_indexes = [i + 1 for i, loss in enumerate(test_losses) if loss == best_test_loss]
    # Calculate the cumulative time up to each best index
    total_times_up_to_best_test = [sum(time_list[:i]) for i in best_test_indexes]

    print("Best Test Loss:", best_test_loss)
    print("Corresponding Train Loss:", train_losses[best_test_indexes[0]])
    print("Indexes of Best Test Loss:", best_test_indexes)
    print("Total Times up to these iterations (seconds):", total_times_up_to_best_test)
    return best_test_loss, total_times_up_to_best_test[0]
