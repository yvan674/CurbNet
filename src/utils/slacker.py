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


class Slacker:
    @staticmethod
    def send_message(subject, message):
        """Sends the message as an slack message with subject header.

        Args:
            subject (str): The subject header.
            message (str): The message body.
        """
        try:
            bot_token = os.environ["SLACK_API_TOKEN"]
        except KeyError:
            return

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

        client = slack.WebClient(token=bot_token)

        # client.chat_postMessage(channel='general', blocks=blocks)
        response = client.chat_postMessage(
            channel="#general",
            blocks=blocks,
        )
