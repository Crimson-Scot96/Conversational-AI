from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.bm25 import retrieve_summary  # Import your BM25 retrieval function
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)


class ActionRouteToBM25(Action):
    def name(self) -> str:
        return "action_route_to_bm25"  # Action name that Rasa will call

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        # Get the user's message (the query)
        query = tracker.latest_message.get('text')

        # Extract the user's intent
        intent = tracker.latest_message['intent'].get('name')

        # Log the query and intent
        logging.debug(f"Received query: {query}")
        logging.debug(f"Detected intent: {intent}")

        try:
            # Pass both the query and intent to the retrieve_summary function
            top_summary = retrieve_summary(query, intent)  # Modify the function call to pass the intent

            # Check if the retrieve_summary function returned "not confident"
            if top_summary == "not confident":
                dispatcher.utter_message(text="Sorry, I couldn't find an answer.")
            else:
                # Send the top-ranked summary to the user
                dispatcher.utter_message(text=top_summary)

        except Exception as e:
            # Handle any exceptions gracefully
            dispatcher.utter_message(text="Sorry, I couldn't find an answer.")
            logging.error(f"Error: {e}")

        return []
