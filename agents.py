import openai
import psycopg2
from psycopg2.extras import RealDictCursor

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class DataIngestionAgent:
    """Agent responsible for fetching employee timing data from PostgreSQL"""

    async def process(self, data: List[Dict] = None) -> Dict[str, Any]:
        """Fetch data from PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                dbname="employee_login",
                user="employee_login_user",
                password="8FYvZbwfP2Bw69GzxX8RRw0sKi7BDpxc",
                host="dpg-d3b77nmmcj7s73em7lhg-a.oregon-postgres.render.com",
                port=5432,
                cursor_factory=RealDictCursor
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employee_timing;")  # Replace with your table name
            records = cursor.fetchall()
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Database fetch error: {e}")

        if not records:
            raise ValueError("No data retrieved from database")

        schema = {
            'fields': list(records[0].keys()) if records else [],
            'record_count': len(records)
        }

        return {
            'status': 'completed',
            'schema': schema,
            'processed_records': len(records),
            'data': records
        }
class ReasoningAgent:
    """Agent for pattern analysis and reasoning using OpenAI GPT"""

    async def process(self, data: List[Dict], analysis_result: Dict) -> Dict[str, Any]:
        """Analyze patterns and generate insights via OpenAI GPT"""
        data_summary = {
            'total_employees': analysis_result['total_records'],
            'average_working_hours': analysis_result['avg_working_hours'],
            'overtime_cases': analysis_result['overtime_count'],
            'undertime_cases': analysis_result['undertime_count'],
            'attendance_rate': analysis_result['attendance_rate']
        }

        prompt = f"""
        You are an HR data analyst. Analyze this employee timing data:

        Data Summary:
        - Total employee records: {data_summary['total_employees']}
        - Average working hours: {data_summary['average_working_hours']:.2f}
        - Overtime cases: {data_summary['overtime_cases']}
        - Undertime cases: {data_summary['undertime_cases']}
        - Overall attendance rate: {data_summary['attendance_rate']:.1f}%

        Identify:
        1. Key patterns in the data
        2. Productivity or scheduling issues
        3. Workforce trends and anomalies
        4. Risk factors for management
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an expert HR analyst."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            llm_analysis = response.choices[0].message.content

            patterns = []
            risks = []
            text_lower = llm_analysis.lower()
            if 'overtime' in text_lower:
                patterns.append('Overtime patterns detected')
            if 'undertime' in text_lower:
                patterns.append('Underutilization patterns identified')
            if 'burnout' in text_lower or 'overwork' in text_lower:
                risks.append('Employee burnout risk')
            if 'compliance' in text_lower:
                risks.append('Compliance risk')

            return {
                'status': 'completed',
                'llm_analysis': llm_analysis,
                'patterns_identified': patterns,
                'risk_factors': risks
            }

        except Exception as e:
            return {
                'status': 'failed',
                'llm_analysis': f'OpenAI API error: {e}',
                'patterns_identified': [],
                'risk_factors': []
            }
class InsightsAgent:
    """Agent for compliance insights using OpenAI GPT"""

    async def process(self, data: List[Dict], analysis_result: Dict, reasoning_result: Dict) -> Dict[str, Any]:
        under_performers = [
            {
                'name': r['name'],
                'date': r['date'],
                'login_time': r['login_time'],
                'logout_time': r['logout_time'],
                'working_hours': r['working_hours'],
                'deficit': round(9 - r['working_hours'], 2)
            }
            for r in analysis_result['timing_records'] if r['working_hours'] < 9
        ]

        prompt = f"""
        You are an HR compliance specialist. Analyze the following under-performers:

        Under-Performers Count: {len(under_performers)}
        Average Working Hours: {analysis_result['avg_working_hours']:.2f}
        Previous Analysis: {reasoning_result.get('llm_analysis', '')}

        Provide:
        1. Specific compliance violations
        2. Individual recommendations for under-performers
        3. Systematic policy improvements
        4. Risk mitigation strategies
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an HR compliance specialist."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            llm_insights = response.choices[0].message.content

            recommendations = [
                {
                    'title': '⏰ Login/Logout Compliance',
                    'description': f'Processed {len(analysis_result["timing_records"])} timing records',
                    'priority': 'High',
                    'action': 'Review individual patterns'
                }
            ]

            return {
                'status': 'completed',
                'under_performers': under_performers,
                'llm_insights': llm_insights,
                'compliance_rate': round((len([r for r in analysis_result['timing_records'] if r['compliant']]) / len(analysis_result['timing_records'])) * 100, 1),
                'recommendations': recommendations
            }

        except Exception as e:
            return {
                'status': 'failed',
                'under_performers': under_performers,
                'llm_insights': f'OpenAI API error: {e}',
                'compliance_rate': 0,
                'recommendations': []
            }
import asyncio
from typing import List, Dict, Any

async def main():
    # Step 1: Fetch data from the database
    ingestion_agent = DataIngestionAgent()
    ingestion_result = await ingestion_agent.process()
    records = ingestion_result['data']
    print(f"Fetched {len(records)} records from the database.\n")

    # Step 2: Basic analysis
    timing_records = []
    total_hours = 0
    overtime_count = 0
    undertime_count = 0
    compliant_count = 0

    for r in records:
        working_hours = r['working_hours']
        total_hours += working_hours
        overtime_count += 1 if working_hours > 9 else 0
        undertime_count += 1 if working_hours < 9 else 0
        compliant = 9 <= working_hours <= 10  # Example compliance check
        if compliant:
            compliant_count += 1

        timing_records.append({**r, 'compliant': compliant})

    total_records = len(records)
    avg_working_hours = total_hours / total_records
    attendance_rate = (compliant_count / total_records) * 100

    analysis_result = {
        'timing_records': timing_records,
        'total_records': total_records,
        'avg_working_hours': avg_working_hours,
        'overtime_count': overtime_count,
        'undertime_count': undertime_count,
        'attendance_rate': attendance_rate
    }

    print(f"Basic Analysis:\n - Average Working Hours: {avg_working_hours:.2f}\n - Overtime Cases: {overtime_count}\n - Undertime Cases: {undertime_count}\n - Compliance Rate: {attendance_rate:.1f}%\n")

    # Step 3: Reasoning via GPT
    reasoning_agent = ReasoningAgent()
    reasoning_result = await reasoning_agent.process(records, analysis_result)
    print(f"Reasoning Analysis:\n{reasoning_result.get('llm_analysis', '')}\nPatterns Detected: {reasoning_result.get('patterns_identified', [])}\nRisk Factors: {reasoning_result.get('risk_factors', [])}\n")

    # Step 4: Compliance insights via GPT
    insights_agent = InsightsAgent()
    insights_result = await insights_agent.process(records, analysis_result, reasoning_result)
    print(f"Compliance Insights:\n{insights_result.get('llm_insights', '')}\n")
    print(f"Recommendations:\n{insights_result.get('recommendations', [])}\n")
        # Print under-performers
    under_performers = insights_result.get('under_performers', [])
    if under_performers:
        print("Under-Performers:")
        for up in under_performers:
            print(f"- {up['name']} on {up['date']} worked {up['working_hours']} hrs (Deficit: {up['deficit']} hrs)")
    else:
        print("No under-performers detected.")

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())



