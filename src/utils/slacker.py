"""Slacker.

Creates messages to send to a slack bot when something is amiss. To make sure
this runs, the following environment variable must be added: SLACK_API_TOKEN.

To add this, follow the instructions at api.slack.com to get a slack api token.
A new app must be created and a bot added to the app. The API token can then
be added to environment variables using whatever method works best for the host
OS.

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
