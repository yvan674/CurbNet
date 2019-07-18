"""Data Processor.

Crunches the numbers and calculates the time left, as well as writes the status
file so we only need to code that once.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import datetime
from utils.slacker import Slacker


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
        finish_at = "NaN"
    else:
        steps_total = float((max_step * max_epoch))
        # Add the validation steps
        steps_total += float(validation_steps * max_epoch)

        # If we're in validation, then we've reached the max step in this
        # epoch + the 10 steps for validation so we add
        # validation * max_step
        steps_done_this_epoch = float(step + 1
                                      + (validation * max_step))

        steps_times_epochs_done = float(max_step * (epoch - 1)
                                        + validation_steps * (epoch - 1))

        running_step_count = steps_done_this_epoch + steps_times_epochs_done

        steps_left = (steps_total - running_step_count)

        time_left = int(steps_left / rate)
        time_left = datetime.timedelta(seconds=time_left)
        finish_at = datetime.datetime.now() + time_left
        finish_at = finish_at.strftime("%a, %d %b, %I:%M:%S %p")

        max_step = validation_steps if validation else max_step

    # Now write the status file
    if step % 10 == 0 or (step == validation_steps
                          and epoch == max_epoch
                          and validation):
        with open(status_file_path, 'w') as status_file:
            lines = ["Step: {}/{}\n".format(step, max_step),
                     "Epoch: {}/{}\n".format(epoch, max_epoch),
                     "Accuracy: {:.3f}%, {:.3f}%, {:.3f}%".format(
                         accuracy[0] * 100., accuracy[1] * 100.,
                         accuracy[2] * 100.),
                     "Loss: {:.3f}\n".format(loss),
                     "Rate: {:.3f} steps/s\n".format(rate),
                     "Time left: {}\n".format(str(time_left)),
                     "Finishes at: {}\n".format(finish_at)
                     ]

            if step == validation_steps and epoch == max_epoch and validation:
                finish_at = datetime.datetime.now()
                finish_at = finish_at.strftime("%a, %d %b, %I:%M:%S %p")
                lines[5] = "Time left: -\n"
                lines[6] = "Finished at: {}".format(finish_at)
                lines.append("Finished training.\n")

                message = "".join(lines)
                Slacker.send_message(message, "Finished Training")

            status_file.writelines(lines)

        if epoch % 10 == 0 and validation and step == validation_steps:
            message = "".join(lines)

            Slacker.send_message(message,
                                 "Update: Finished epoch {}".format(epoch))

    return time_left, running_step_count, steps_total
