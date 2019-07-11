"""Data Processor.

Crunches the numbers and calculates the time left, as well as writes the status
file so we only need to code that once.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from datetime import timedelta


def process_data(step, epoch, accuracy, loss, rate, status_file_path,
                 validation, max_step, max_epoch, validation_steps):
    """Processes the time left and writes the status file.

    Returns:
        str: Time left as a human readable string.
        float: Running step count.
        float: Total number of steps the trainer will iterate through.
    """
    # Calculate time left
    steps_total = 1
    running_step_count = 0
    if rate == 0:
        time_left = "NaN"
    else:
        steps_total = float((max_step * max_epoch))
        # Add the validation steps
        steps_total += float(validation_steps * max_epoch)

        # If we're in validation, then we've reached the max step in this
        # epoch + the 10 steps for validation so we add
        # validation * max_step
        steps_done_this_epoch = float(step + 1
                                      + (validation * max_step))

        steps_times_epochs_done = float(max_step * (epoch - 1))

        running_step_count = steps_done_this_epoch + steps_times_epochs_done

        steps_left = (steps_total - running_step_count)

        time_left = int(steps_left / rate)
        time_left = str(timedelta(seconds=time_left))

    # Now write the status file
    if step % 10 == 0 or (step == validation_steps
                          and epoch == max_epoch
                          and validation):
        with open(status_file_path, 'w') as status_file:
            lines = ["Step: {}/{}\n".format(step, max_step),
                     "Epoch: {}/{}\n".format(epoch, max_epoch),
                     "Accuracy: {:.2f}%\n".format(accuracy * 100),
                     "Loss: {:.3f}\n".format(loss),
                     "Rate: {:.3f} steps/s\n".format(rate),
                     "Time left: {}\n".format(time_left)]

            if step == validation_steps and epoch == max_epoch and validation:
                lines[5] = "Time left: -\n"
                lines.append("Finished training.\n")

            status_file.writelines(lines)

    return time_left, running_step_count, steps_total
