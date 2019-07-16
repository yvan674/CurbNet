"""Slacker.

Creates messages to send to a slack bot when something is amiss.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import slack
import os
import ssl as ssl_lib
import certifi


class Slacker:
    @staticmethod
    def send_message(subject, message):
        """Sends the message as an slack message with subject header.

        Args:
            subject (str): The subject header.
            message (str): The message body.
        """
        bot_token = "xoxb-698100624807-696255003584-ek6Oqxm6c5ugHZRwuJhFKSCN"

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
                        "text": "```" + message + "```"
                    }
                }
            ]

        client = slack.WebClient(token=bot_token)

        # client.chat_postMessage(channel='general', blocks=blocks)
        response = client.chat_postMessage(
            channel="#general",
            blocks=blocks,
        )
