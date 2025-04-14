
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# might connect to sbert idk
class ActionRouteToSBERT(Action):
    def name(self) -> Text:
        return "action_route_to_sbert"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
        ) -> List[Dict[Text, Any]]:

        user_input = tracker.latest_message.get('text')
        # testing output
        answer = f"[Placeholder response] You said: '{user_input}'"

        dispatcher.utter_message(text=answer)
        return []
