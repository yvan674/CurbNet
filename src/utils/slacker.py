"""Slacker.

Creates messages to send to a slack bot when something is amiss.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
try:
    import slack
except ImportError:
    slack = None
import os
import datetime


class Slacker:
    @staticmethod
    def send_code(subject, message):
        """Sends a code-formatted message.

        Args:
            subject (str): The subject header.
            message (str): The message body, which will be sent as code
                formatting.
        """

        message = "```\n{}\n```".format(message)

        blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*" + subject + "*"
                    }
                 },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]

        Slacker._send_blocks(blocks)

    @staticmethod
    def send_message(message, header=None):
        """Sends a regular message, with an optional header.

        Args:
             message (str): The message to be sent.
             header (str): The header, which will be added to the message in
                bold. Defaults to None.
        """
        # Try to get the bot token
        blocks = []
        if header:
            blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*" + header + "*"
                    }
                 })
            blocks.append({"type": "divider"})

        blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                })
        Slacker._send_blocks(blocks)

    @staticmethod
    def _send_blocks(blocks):
        """Sends preformatted blocks to slack."""
        try:
            bot_token = os.environ["SLACK_API_TOKEN"]
        except KeyError:
            return
        client = slack.WebClient(token=bot_token)

        # client.chat_postMessage(channel='general', blocks=blocks)
        response = client.chat_postMessage(
            channel="#general",
            blocks=blocks,
        )


if __name__ == '__main__':
    time_left = datetime.timedelta(days=2)
    finish_at = datetime.datetime.now() + time_left
    finish_at = finish_at.strftime("%a, %d %b, %I:%M:%S %p")

    lines = ["Step: {}/{}\n".format(5, 472),
             "Epoch: {}/{}\n".format(2, 200),
             "Accuracy: {:.2f}%\n".format(.88521 * 100),
             "Loss: {:.3f}\n".format(.2125),
             "Rate: {:.3f} steps/s\n".format(8.2574),
             "Time left: {}\n".format(str(time_left)),
             "Finishes at: {}\n".format(finish_at)
             ]

    output_text = ""
    for line in lines:
        output_text += line

    print(output_text)
    Slacker.send_message(output_text,
                         "Update: Finished epoch {}".format(1))