from locust import HttpUser, task, between
import random

MESSAGES = [
    "mera card kho gaya hai block kardo 453289012345",
    "paise transfer kiye the recipient ko nahi mile",
    "card kab tak deliver hoga tracking bata do",
    "statement me extra charges lag gaye hai dispute create karo",
    "mera card kaam nahi kar raha pos machine par",
    "transaction declined ho gaya par paise kat gaye"
]

class SupportUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def triage_message(self):
        msg = random.choice(MESSAGES)
        self.client.post("/triage", json={"message": msg})