"""
gen_data.py: synthetic sample-data generator for the IAPA CA.

Produces two CSVs (rule/template-based, SEEDED, fully reproducible, NOT real
Globacom data, GDPR-safe, declared as synthetic in the AI-usage sheet):

  1. data/tickets.csv    -> feeds the Multinomial Naive Bayes triage bot
                            (ticket_text -> category -> routed team)
  2. data/customers.csv  -> feeds the RPA "Daily Customer Creation" report bot

Run:  python3 gen_data.py
"""

import csv
import random
from datetime import datetime, timedelta

from Config import SEED, ROWS_PER_CLASS, N_CUSTOMERS, N_DIRTY, TICKETS_PATH, CUSTOMERS_PATH, ROUTING

random.seed(SEED)

#ticket templates
#15 phrasings per class, written in the mixed voice real Glo customers use:
#formal English, pidgin, and the occasional all-caps rant. More varied phrasing
#better Naive Bayes generalisation. Total rows stay at 42 x 5 = 210.

#slot values the generator drops into the {placeholders}
SLOTS = {
    "location": ["Lekki", "Ikeja", "Surulere", "Yaba", "Ajah", "Ikorodu",
                 "Victoria Island", "Ibadan", "Abuja", "Port Harcourt"],
    "amount":   ["N500", "N1000", "N1500", "N2000", "N5000"],
    "product":  ["6GB weekly bundle", "12GB monthly plan", "Glo Yakata offer",
                 "Special Data bundle", "N1000 airtime top-up"],
}

#15 phrasings per class. the {slots} get filled at generation time.
TEMPLATES = {
    #1.SIM & Activation  
    "SIM_ACTIVATION": [
        "My new SIM {msisdn} is not registering on the network after activation.",
        "I did SIM swap yesterday but the new SIM still shows no service.",
        "Please activate my line {msisdn}, I bought it in {location} but it is not working.",
        "SIM not showing Glo network since I inserted it, tried restarting, still nothing.",
        "I ported to Glo but my SIM {msisdn} has been blank for two days.",
        "New SIM says 'SIM not provisioned', I registered it at the Glo shop in {location}.",
        "My old SIM stopped working after the swap, and the new one won't come on.",
        "Abeg I bought this SIM for {location} since last week, e never activate till now.",
        "Registered my SIM with NIN at the shop but the line {msisdn} is still not active.",
        "They told me activation takes 24 hours, it is now 4 days and {msisdn} is still dead.",
        "PLEASE ACTIVATE MY SIM, I have registered it TWICE already and nothing.",
        "Welcome SMS never came after I inserted the new SIM, is my line {msisdn} activated?",
        "My SIM got blocked after I lost my phone, I did replacement in {location} but the new one is not connecting.",
        "The agent for {location} do the swap but my number never port enter the new SIM.",
        "Inserted SIM shows emergency calls only, I finished registration yesterday.",
    ],

    #2. Network & Connectivity
    "NETWORK": [
        "No signal in {location} since this morning, my calls keep dropping.",
        "Data is extremely slow on {msisdn}, cannot even open WhatsApp in {location}.",
        "Network has been down in {location} for hours, nobody can call me.",
        "Calls are not connecting, it says 'network busy' every time I dial.",
        "I have full bars but no internet, this started after the {product} subscription.",
        "Constant call drops in {location}, very poor network for the past three days.",
        "My phone shows E instead of 4G in {location}, browsing is impossible.",
        "Glo network don disappear for {location} since NEPA take light yesterday.",
        "I cannot receive calls, people say my line {msisdn} is switched off but my phone is on.",
        "Browsing is dead in {location}, even Google will not open, is there an outage?",
        "THREE DAYS NOW no network in my area {location}, how are we supposed to work?",
        "Voice calls keep breaking, the person cannot hear me clearly on {msisdn}.",
        "SMS is not delivering, messages hang on sending since yesterday evening.",
        "The network for {location} na one bar today, video call no dey possible at all.",
        "4G keeps switching to 3G every few minutes in {location} and downloads restart.",
    ],

    #3. Payment / Billing
    "PAYMENT_BILLING": [
        "I paid {amount} for the {product} but I was debited and got no data.",
        "My airtime of {amount} disappeared right after I recharged {msisdn}.",
        "I subscribed to the {product} but it charged me {amount} twice.",
        "Wrong deduction on my line {msisdn}, {amount} was removed for nothing.",
        "I bought {amount} airtime at {location} but it never reflected on my line.",
        "Bill is showing {amount} extra that I did not use, please check my account.",
        "I was charged {amount} for a {product} I never activated.",
        "Why did Glo remove {amount} from my airtime for caller tune I never requested?",
        "Recharged {amount} through my bank app, money left my account but no airtime on {msisdn}.",
        "Dem don deduct my {amount} again for one auto-renewal wey I no subscribe to.",
        "My data finished too fast, {amount} worth of {product} gone in one day, check my usage.",
        "STOP DEDUCTING MY AIRTIME. {amount} vanished today again on {msisdn}.",
        "The ussd code charged me {amount} but the {product} did not activate.",
        "I loaded a recharge card of {amount} for {location} and it said invalid but the card is new.",
        "Every morning una dey remove small small money from my line {msisdn}, wetin be that?",
    ],

    #4. Refunds
    "REFUNDS": [
        "I was double-charged {amount} for the {product}, I need a refund.",
        "Please reverse the {amount} wrongly deducted from {msisdn} yesterday.",
        "My {amount} recharge failed but the money left my bank, refund me please.",
        "I want a refund of {amount}, the {product} did not work at all.",
        "Debited {amount} twice for one subscription, kindly return one.",
        "The {product} was cancelled but my {amount} has not been refunded.",
        "Send back my {amount}, the data bundle never entered my line {msisdn}.",
        "It has been two weeks since I reported the failed recharge, where is my {amount} refund?",
        "Abeg return my {amount}, the transfer to the wrong number was your ussd error not mine.",
        "I need my money back, {amount} was taken for the {product} and the sub never worked.",
        "REFUND MY {amount} NOW, this is the third time I am writing about this.",
        "Customer care promised a reversal of {amount} on {msisdn} last Friday, nothing yet.",
        "The failed transaction of {amount} should be reversed to my main balance please.",
        "Una charge me twice for the same {product}, I want one of the {amount} back.",
        "My refund reference from the {location} shop has still not been paid, it is {amount}.",
    ],

    #5. General / Other  (default catch-all -> "type 1")
    "GENERAL": [
        "My phone is not working, please help.",
        "I need help with my Glo line {msisdn}.",
        "How do I check my data balance?",
        "I want to change my tariff plan, what do I do?",
        "Please I have a complaint about my account.",
        "Can someone call me back about my line {msisdn}?",
        "I don't understand the last message Glo sent me.",
        "Good afternoon, how can I stop these promo messages you send me every day?",
        "Which code do I use to borrow airtime on {msisdn}?",
        "Abeg how person go take transfer data give another Glo line?",
        "I want to know the difference between {product} and the normal plan.",
        "How do I link my NIN to this line, is there a code for it?",
        "My complaint from last week has no update, ticket was for line {msisdn}.",
        "Do you people have a Glo shop in {location}? I need to sort something out.",
        "Wetin be the code to check how much data remain for my line?",
    ],
}

#helpers
NIGERIAN_FIRST = ["Taiwo", "Chidi", "Aisha", "Emeka", "Ngozi", "Kunle", "Fatima",
                  "Ifeoma", "Segun", "Blessing", "Yusuf", "Amaka", "Tunde",
                  "Halima", "Obinna", "Zainab", "Femi", "Chioma", "Bello", "Uche"]
NIGERIAN_LAST  = ["Okafor", "Adeyemi", "Bello", "Eze", "Ibrahim", "Balogun",
                  "Nwosu", "Abubakar", "Ogunleye", "Chukwu", "Musa", "Okonkwo",
                  "Adebayo", "Suleiman", "Anyanwu", "Lawal"]
PREFIXES = ["0805", "0705", "0815", "0905", "0807", "0811"]
CHANNELS = ["Call Center", "Email", "USSD", "Retail Shop", "Social Media"]
SEGMENTS = ["Prepaid", "Postpaid", "Corporate", "SME"]
STATUSES = ["Active", "Pending", "Suspended"]


def make_msisdn():
    """A realistic 11-digit Glo number: prefix + 7 random digits."""
    return random.choice(PREFIXES) + "".join(random.choice("0123456789") for _ in range(7))


def fill(template):
    """Drop slot values into a template. Templates without slots pass through."""
    return template.format(
        msisdn=make_msisdn(),
        location=random.choice(SLOTS["location"]),
        amount=random.choice(SLOTS["amount"]),
        product=random.choice(SLOTS["product"]),
    )


def rand_datetime(days_back):
    """A timestamp within the last `days_back` days, during business hours."""
    base = datetime(2026, 7, 7, 9, 0, 0)
    delta = timedelta(days=random.randint(0, days_back),
                      hours=random.randint(0, 8),
                      minutes=random.randint(0, 59))
    return (base - delta).strftime("%Y-%m-%d %H:%M:%S")


#tickets.csv
def build_tickets(path):
    rows = []
    tid = 1
    for category in ROUTING:                         #5 classes (keys of the routing map)
        for _ in range(ROWS_PER_CLASS):              #42 each
            template = random.choice(TEMPLATES[category])
            rows.append({
                "ticket_id":   f"TCK{tid:05d}",
                "msisdn":      make_msisdn(),
                "created_at":  rand_datetime(days_back=30),
                "channel":     random.choice(CHANNELS),
                "ticket_text": fill(template),
                "category":    category,
            })
            tid += 1
    random.shuffle(rows)                             #mix classes so it's not sorted
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


#customers.csv
def build_customers(path):
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        rows.append({
            "CustomerID":      f"CUST{i:06d}",
            "FirstName":       random.choice(NIGERIAN_FIRST),
            "LastName":        random.choice(NIGERIAN_LAST),
            "MSISDN":          make_msisdn(),
            "CustomerSegment": random.choice(SEGMENTS),
            "CreationDate":    rand_datetime(days_back=7).split(" ")[0],  #date only
            "Status":          random.choice(STATUSES),
        })

    #inject N_DIRTY deliberately broken records, real mediation dumps are never
    #perfectly clean, and the RPA bot's validator needs something to catch
    breakers = [
        lambda r: r.update(MSISDN="0999" + r["MSISDN"][4:]),        #prefix not in catalogue
        lambda r: r.update(MSISDN=r["MSISDN"][:7]),                 #too short, not 11 digits
        lambda r: r.update(CustomerSegment="VIP"),                  #segment catalogue doesn't know
        lambda r: r.update(FirstName=""),                           #required field empty
        lambda r: r.update(CreationDate="07/07/2026"),              #wrong date format
    ]
    dirty_rows = random.sample(rows, N_DIRTY)
    for row, breaker in zip(dirty_rows, breakers):
        breaker(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    t = build_tickets(TICKETS_PATH)
    c = build_customers(CUSTOMERS_PATH)
    print(f"{TICKETS_PATH}   -> {t} rows ({len(ROUTING)} classes x {ROWS_PER_CLASS})")
    print(f"{CUSTOMERS_PATH} -> {c} rows")
    print("SEED =", SEED, "(reproducible). Synthetic data, declare in AI-usage sheet.")
