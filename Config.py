#Config.py: every path, as well as constant for both bots in one place (single source of truth)
#shared folders
DATA_DIR      = 'data/'
ARTIFACTS_DIR = 'artifacts/'
OUTPUTS_DIR   = 'outputs/'

#data files
TICKETS_PATH     = 'data/tickets.csv'       #training data for the triage bot
NEW_TICKETS_PATH = 'data/new_tickets.csv'   #unseen tickets the bot routes
CUSTOMERS_PATH   = 'data/customers.csv'     #input for the RPA report bot

#synthetic data generator
SEED = 42                  #fixed; same output every run makes it reproducible
ROWS_PER_CLASS = 42        #42 x 5 classes = 210 tickets
N_CUSTOMERS = 120
N_DIRTY = 5                #deliberately broken records so the validator has work to do

#triage bot (AI / agentic)
#the text column the model reads, and the label column it predicts
TICKET_TEXT = 'ticket_text'
CATEGORY = 'category'
#columns we expect to exist in the training csv
REQUIRED_TICKET_COLUMNS = [TICKET_TEXT, CATEGORY]
#my 'i'm not sure, ask a human' line; below this the ticket gets flagged
CONF_THRESHOLD = 0.55
#where the joblib artifacts live
VECTORIZER_PATH = 'artifacts/vectorizer.joblib'
MODEL_PATH = 'artifacts/model_nb.joblib'
#where routed tickets get written
ROUTED_TICKETS_PATH = 'outputs/routed_tickets.csv'
#predicted category -> backend team the ticket gets routed to
ROUTING = {
    "SIM_ACTIVATION":  "SIM Team",
    "NETWORK":         "Network Ops",
    "PAYMENT_BILLING": "Billing Team",
    "REFUNDS":         "Finance / Refunds",
    "GENERAL":         "Tier-1 Desk",
}

#RPA report bot
#columns every customer record must have (Customer-domain data dictionary)
REQUIRED_CUSTOMER_COLUMNS = ['CustomerID', 'FirstName', 'LastName', 'MSISDN',
                             'CustomerSegment', 'CreationDate', 'Status']
#field-catalogue validation rules (mirrors the Glo report field catalogue idea)
MSISDN_LENGTH  = 11
VALID_PREFIXES = ["0805", "0705", "0815", "0905", "0807", "0811"]
VALID_SEGMENTS = ["Prepaid", "Postpaid", "Corporate", "SME"]
VALID_STATUSES = ["Active", "Pending", "Suspended"]
#where the finished report + rejected records land
REPORT_XLSX_PATH = 'outputs/daily_customer_creation_report.xlsx'
REPORT_DOCX_PATH = 'outputs/daily_customer_creation_report.docx'
EXCEPTIONS_PATH  = 'outputs/validation_exceptions.csv'
