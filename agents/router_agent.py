import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import llm
from agents.academic_agent import academic_agent
from agents.admission_agent import admission_agent
from agents.placement_agent import placement_agent
from agents.hostel_agent import hostel_agent
from agents.fee_agent import fee_agent
from agents.campus_agent import campus_agent
from agents.department_agent import department_agent
from agents.notice_agent import notice_agent


def router_agent(question):

    router_prompt = f"""
You are an AI router for a university assistant.

Your job is to choose the correct department that should answer the question.

Possible categories:

ACADEMIC → syllabus, attendance, exam rules, course details  
ADMISSION → admission process, eligibility, documents required  
PLACEMENT → highest package, average package, companies, internships  
HOSTEL → hostel rules, curfew time, mess, hostel facilities  
FEE → tuition fee, hostel fee, payment structure  
CAMPUS → campus facilities, library, sports, labs  
DEPARTMENT → department details, faculty, programs  
NOTICE → announcements, circulars, notices  

Return ONLY ONE WORD from the list above.

Question:
{question}
"""

    decision = llm.invoke(router_prompt).strip().upper()

    print("Router Decision:", decision)

    if "ACADEMIC" in decision:
        return academic_agent(question)

    elif "ADMISSION" in decision:
        return admission_agent(question)

    elif "PLACEMENT" in decision:
        return placement_agent(question)

    elif "HOSTEL" in decision:
        return hostel_agent(question)

    elif "FEE" in decision:
        return fee_agent(question)

    elif "CAMPUS" in decision:
        return campus_agent(question)

    elif "DEPARTMENT" in decision:
        return department_agent(question)

    elif "NOTICE" in decision:
        return notice_agent(question)

    else:
        return academic_agent(question)