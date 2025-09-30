import os
import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import psycopg2
from psycopg2.extras import RealDictCursor

# -------------------- Config --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

# -------------------- Agents --------------------
class DataIngestionAgent:
    async def process(self) -> Dict[str, Any]:
        """Fetch data from PostgreSQL employees table"""
        try:
            conn = psycopg2.connect(cursor_factory=RealDictCursor, **DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employees;")
            records = cursor.fetchall()
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Database fetch error: {e}")

        if not records:
            raise ValueError("No data retrieved from database")

        # Auto-compute working hours if login/logout exist
        for rec in records:
            try:
                login = rec.get("login_time")
                logout = rec.get("logout_time")
                if login and logout:
                    fmt = "%H:%M:%S"
                    t_in = datetime.strptime(str(login), fmt)
                    t_out = datetime.strptime(str(logout), fmt)
                    rec["Working hours"] = round((t_out - t_in).total_seconds()/3600, 2)
                else:
                    rec["Working hours"] = 0
            except:
                rec["Working hours"] = 0

        schema = {'fields': list(records[0].keys()), 'record_count': len(records)}
        return {'status': 'completed', 'schema': schema, 'processed_records': len(records), 'data': records}


class AnalysisAgent:
    async def process(self, data: List[Dict]) -> Dict[str, Any]:
        working_hours = []
        timing_records = []

        for record in data:
            hours = float(record.get('Working hours', 0))
            working_hours.append(hours)
            timing_records.append({
                'name': record.get('name', 'Unknown'),
                'date': str(record.get('date')),
                'working_hours': hours,
                'compliant': hours >= 9,
                'deficit': max(0, 9 - hours)
            })

        avg_hours = sum(working_hours)/len(working_hours)
        overtime = len([h for h in working_hours if h > 8])
        undertime = len([h for h in working_hours if h < 8])

        return {
            'total_records': len(data),
            'avg_working_hours': avg_hours,
            'overtime_count': overtime,
            'undertime_count': undertime,
            'timing_records': timing_records
        }


class ReasoningAgent:
    async def process(self, data: List[Dict], analysis_result: Dict) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following employee timing data:
        Total employees: {analysis_result['total_records']}
        Avg working hours: {analysis_result['avg_working_hours']:.2f}
        Overtime cases: {analysis_result['overtime_count']}
        Undertime cases: {analysis_result['undertime_count']}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an HR analyst."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            llm_analysis = response.choices[0].message.content
        except Exception as e:
            llm_analysis = f"OpenAI error: {e}"

        return {'status': 'completed', 'llm_analysis': llm_analysis}


class InsightsAgent:
    async def process(self, data: List[Dict], analysis_result: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        under_performers = [r for r in analysis_result['timing_records'] if r['working_hours'] < 9]
        compliance_rate = round(len([r for r in analysis_result['timing_records'] if r['compliant']])/len(data)*100,1)
        return {
            'under_performers': under_performers,
            'compliance_rate': compliance_rate,
            'llm_insights': reasoning_result['llm_analysis']
        }

# -------------------- Visualization --------------------
def create_visualizations(analysis_result: Dict, insights_result: Dict):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hours = [r['working_hours'] for r in analysis_result['timing_records']]
    compliant = sum([1 for r in analysis_result['timing_records'] if r['compliant']])
    non_compliant = len(hours) - compliant

    # Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(hours, bins=10, kde=True, color='skyblue')
    plt.title("Distribution of Working Hours")
    plt.xlabel("Hours")
    plt.ylabel("Employees")
    plt.savefig(f"working_hours_{date_str}.png")
    plt.close()

    # Pie chart
    plt.figure(figsize=(6,6))
    plt.pie([compliant, non_compliant], labels=["Compliant","Non-Compliant"], autopct='%1.1f%%', colors=['green','red'])
    plt.title("Compliance Rate")
    plt.savefig(f"compliance_pie_{date_str}.png")
    plt.close()

    # Top underperformers
    under = sorted(insights_result['under_performers'], key=lambda x: x['deficit'], reverse=True)[:10]
    if under:
        plt.figure(figsize=(8,5))
        sns.barplot(x=[u['name'] for u in under], y=[u['deficit'] for u in under], palette='Reds_r')
        plt.title("Top 10 Under-Performers by Hours Deficit")
        plt.ylabel("Hours Short")
        plt.xlabel("Employee")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"top_underperformers_{date_str}.png")
        plt.close()

# -------------------- Agent Workflow --------------------
class AgentWorkflow:
    def __init__(self):
        self.data_agent = DataIngestionAgent()
        self.analysis_agent = AnalysisAgent()
        self.reasoning_agent = ReasoningAgent()
        self.insights_agent = InsightsAgent()

    async def execute_workflow(self, payload=None):
        """
        Optionally accept a payload from Zapier/Slack.
        If payload is None, fallback to fetching from DB.
        """
        if payload:
            # Use the payload as a single-record dataset
            data = [{
                "name": payload.get("user", {}).get("real_name") or "Unknown",
                "date": payload.get("ts_time", None),
                "login_time": payload.get("login_time", None),
                "logout_time": payload.get("logout_time", None),
                "working_hours": None
            }]
            ingestion_result = {"status": "completed", "data": data, "processed_records": 1, "schema": {"fields": list(data[0].keys()), "record_count":1}}
        else:
            # Default: fetch all data from DB
            ingestion_result = await self.data_agent.process()
            data = ingestion_result['data']

        # Analysis → Reasoning → Insights
        analysis_result = await self.analysis_agent.process(data)
        reasoning_result = await self.reasoning_agent.process(data, analysis_result)
        insights_result = await self.insights_agent.process(data, analysis_result, reasoning_result)

        # Visualizations
        create_visualizations(analysis_result, insights_result)

        # Save final results JSON
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_results_file = f"employee_analysis_results_{date_str}.json"
        final_results = {
            "ingestion": ingestion_result,
            "analysis": analysis_result,
            "reasoning": reasoning_result,
            "insights": insights_result
        }
        with open(final_results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"✅ Analysis Complete → Saved '{final_results_file}'")
        return final_results
