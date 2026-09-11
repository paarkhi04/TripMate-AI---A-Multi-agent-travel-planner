from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

user_iput = input("enter travel request:")
response = run_travel_agent(
    user_input = user_iput,
    thread_id ="test_user"
)

print("\nFINAL ANSWER:\n")

print(response["answer"])

#res = search_flights("Plan a 7 days Japan trip from Bangladesh")
# print(res)

#res = tavily_search("best hotels in india")
#print(res)