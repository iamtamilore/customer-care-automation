# Customer Care Process Automation

Two bots that took over two jobs I used to do by hand.

## The story

Every morning on the customer-care desk, two things had to happen before anyone could
start actual work.

**First**, someone pulled the overnight data dump and built the daily report - how many
new customers signed up, broken down by type. Management needed it to know the business
was alive and the numbers were real. Building it meant reading a rulebook spreadsheet and
checking every single field by eye. Is this phone number 11 digits? Is that a real network
prefix? Is this date an actual date? Hundreds of rows, every morning, forever.

**Second**, someone read the customer complaints that came in overnight and decided, one
by one, which team each belonged to. SIM problem? Billing? Refund?

Both jobs were slow, dull, and exactly the kind of thing a tired human gets wrong at 8am.
So I replaced both.

## Why two different bots, and not one

This is the part that matters, and it is the reason the project exists.

Checking whether a phone number has 11 digits is **a rule**. There is one right answer, it
never changes, and no judgement is involved. A computer follows the rule perfectly, forever,
without getting bored. That is a job for **RPA** - a bot that just does the steps.

But reading *"person comot my N2000"* and knowing it means "someone took two thousand
naira off my account, this is a billing issue" - that is not a rule. That is understanding
language. No amount of if-statements gets you there, because people write complaints in
formal English, in Pidgin, in furious all-caps, and they all mean the same thing. That is
a job for **AI**.

So before writing a line of bot code, I mapped the whole process out in BPMN (a standard
way of drawing how work actually flows). The map is what told me which tasks were rules
and which needed judgement. **Map first, automate second.** Automating before you
understand the process just gets you a faster mess.

## The most important knob: knowing when to shut up

The AI bot does not act on everything it thinks.

Every time it classifies a complaint, it also reports how sure it feels, from 0 to 1. If
that number falls below **0.55**, the bot does not route the ticket. It hands it to a human
and says, in effect, *"I think this is a SIM issue, but I am not confident enough. You
look."*

Here is the moment that convinced me this was the right design. I gave the bot five
complaints it had never seen. It labelled **all five correctly** - and still flagged two of
them for a human, because those two only scored 0.37 and 0.41.

It was right, and it did not know it was right.

That gap is the whole point. **Being correct and being confident are different things**,
and a system that cannot tell the difference will eventually route something important to
the wrong place with total conviction. The threshold is the bot admitting it has limits.

## How well does it actually work

**The AI triage bot - 95.2% accurate** on 42 complaints it had never seen.

| Category | Precision | Recall | F1 | Tickets |
|---|---|---|---|---|
| General | 1.00 | 1.00 | 1.00 | 9 |
| Network | 1.00 | 1.00 | 1.00 | 8 |
| SIM activation | 1.00 | 1.00 | 1.00 | 8 |
| Refunds | 0.89 | 0.89 | 0.89 | 9 |
| Payment/Billing | 0.88 | 0.88 | 0.88 | 8 |
| **Overall** | | | **0.952** | **42** |

Three categories are perfect. The bot only ever confuses **billing with refunds** - one
each way, out of 42.

Look at why:

- Billing: *"I was charged N2000 for data I never got"*
- Refund: *"Refund my N2000, I was charged for data I never got"*

Same words. Same amount. Same complaint, almost. A tired human mixes these up too. It is
not a bug in the model - it is the boundary genuinely being blurry, and I would rather
report that honestly than pretend otherwise.

**The RPA bot - caught every single fault.** Out of 120 customer records, 115 were clean
and 5 were broken. It caught all 5, tagged each with exactly which rule it broke, and wrote
them to an exceptions file a human can act on.

Nothing fails silently. That is deliberate: a bot that quietly drops bad rows is worse than
no bot, because now your numbers are wrong *and* nobody knows.

## The five rules the RPA bot enforces

Straight from the real field catalogue:

1. **Nothing important is blank** - no missing name, no missing ID
2. **Phone number is 11 digits and starts with a real network prefix** - 0805 yes, 0999 no
3. **Customer type is one we recognise** - Prepaid, Postpaid, Corporate, SME. Not "VIP"
4. **Account status is real** - Active, Pending, Suspended
5. **The date is an actual date** - `2026-07-07`, not `07/07/2026`

Pass all five, you go in the report. Fail any one, you go in the exceptions file with a
note saying which.

## About the data - read this bit

**No real customer data is used here, and that is on purpose.**

Real telecom mediation and customer records are confidential and GDPR-protected. So I wrote
a generator that produces fake data with the *same shape* as the real thing: 210 complaints
across 5 categories, written in the actual voice customers use (some formal, some Pidgin,
some angry all-caps), plus 120 customer records with exactly 5 deliberately broken - one
per rule, so the validator has something real to catch.

Every number in this README comes from actually running the pipeline. It is all seeded
(`seed=42`), so if you run it, you get the identical result. No hand-waving.

## Try it yourself

```bash
pip install -r requirements.txt

python3 gen_data.py            # 1. make the fake data
python3 triage_bot.py train    # 2. teach the AI bot, save its "brain" to disk
python3 triage_bot.py route    # 3. sort new complaints, flag the doubtful ones
python3 rpa_report_bot.py      # 4. validate records, write the report + exceptions
```

Order matters - step 3 needs the brain that step 2 saves.

## Under the hood, briefly

**TF-IDF** turns each complaint into numbers, because a computer cannot do maths on words.
It weights rare, telling words like "refund" and "prefix" far above common ones like "the"
and "my" - a sort of word-importance scale.

**Multinomial Naive Bayes** reads those numbers and picks the most likely category. It is a
deliberately simple, well-understood model: cheap to train, strong on sparse text, and you
can explain exactly why it decided what it decided. For a system that routes real customer
problems, being explainable beats being fancy.

The vectoriser learns its vocabulary from the **training data only**, then gets frozen. The
model never sees a test word while it is learning - otherwise the 95.2% would be a lie.

## What I would fix next

The test set is 42 tickets. That is small, and I am not going to dress it up as more than
it is. With more data the billing/refund boundary is the first thing I would attack -
probably by giving the model features about *intent* ("give me back" vs "why was I
charged") rather than just words, since the vocabulary genuinely overlaps.

## Files

| Path | What it does |
|---|---|
| `Config.py` | one place for every setting - paths, the 5 rules, routing map, the 0.55 threshold |
| `gen_data.py` | builds the seeded fake dataset |
| `triage_bot.py` | the AI bot - train, evaluate, route |
| `rpa_report_bot.py` | the rule bot - read, validate, reconcile, write |
| `plot_results.py` | regenerates the charts |
| `diagrams/` | the BPMN process maps (before and after) and result figures |
| `artifacts/` | the saved model and vectoriser |
| `outputs/` | routed tickets, exceptions, the daily report |

---

Built for **H9IAPA - Intelligent Agents and Process Automation**, MSc Artificial
Intelligence, National College of Ireland. Based on real customer-care reporting work,
rebuilt from scratch on synthetic data.

**Stack:** Python, scikit-learn, pandas, openpyxl, python-docx, joblib, BPMN 2.0
