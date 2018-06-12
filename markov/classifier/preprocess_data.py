#import unicodecsv as csv
import csv
import wikipedia
import pprint
import urllib
import urllib.request
import time
from tqdm import tqdm

author_dict = {}

with open("pol_accounts.csv", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=";")
    for row in reader:
        author_dict[int(row["id"])] = ([row["screen_name"], row["description"]])

def get_search_string(representative):
    return representative[0] + " " + representative[1]

binding_list = []
with open("pol_bindings.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        binding_list.append(int(row["id"]))

def get_search_string(representative):
    return representative[0] + " " + representative[1]

with open("pol_bindings.csv",'a') as file:
    writer = csv.DictWriter(file, fieldnames=["id","screen_name","democrat","republican"])
    #writer.writeheader()
    for row in tqdm(author_dict):
        if not row in binding_list:
            time.sleep(3)
            user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
            url = "https://www.google.nl/search?q=wikipedia+" + author_dict[row][0]
            headers={'User-Agent':user_agent,}

            request=urllib.request.Request(url,None,headers)
            response = urllib.request.urlopen(request)
            data = str(response.read())
            responses = str(data).split("wikipedia.org/wiki/")[1:]
            democrat = -1
            republican = -1
            if len(responses)>0:
                handle = wikipedia.search(responses[0].split("&")[0])
                if len(handle)>0:
                    try:
                        page = wikipedia.page(handle[0])#
                        searchpage = page.content.lower()

                        democrat = searchpage.count('democrat')
                        republican = searchpage.count('republic')

                        if(searchpage.find('democrat')<searchpage.find('republican')):
                            democrat = democrat + 100

                        if(searchpage.find('democrat')>searchpage.find('republican')):
                            republican = republican + 100
                    except Exception:
                        pass
            writer.writerow({"id": row, "screen_name": author_dict[row][0], "democrat": democrat, "republican": republican})





# from googleapiclient.discovery import build
#
# service = build("customsearch", "v1",
#         developerKey="AIzaSyD1QS165pR5x2VmMGjneXL3zERNW8WdJCY")
#
# #001432505923933560789:cbh8sypy4h4
# #017576662512468239146:omuauf_lfve
# res = service.cse().list(q='bobbyscott', cx='017576662512468239146:omuauf_lfve',).execute()
# pprint.pprint(res)


    # Sleutel 1: 93a63c2d291f4148a548bd4741b4d833
    #
    # Sleutel 2: 3894e4f9af4b483aa730dc34c42f6610
####For Web Results:
#
#
# from py_ms_cognitive import PyMsCognitiveWebSearch
# search_term = "Python Software Foundation"
# search_service = PyMsCognitiveWebSearch('93a63c2d291f4148a548bd4741b4d833', search_term)
#
# first_fifty_result = search_service.search(limit=50, format='json') #1-50
# >>> second_fifty_resul t= search_service.search(limit=50, format='json') #51-100
#
# >>> print (second_fifty_result[0].snippet)
#     u'Python Software Foundation Home Page. The mission of the Python Software Foundation is to promote, protect, and advance the Python programming language, and to ...'
# >>> print (first_fifty_result[0].__dict__.keys()) #see what variables are available.
# ['name', 'display_url', 'url', 'title', 'snippet', 'json', 'id', 'description']
#
#     # To get individual result json:
# >>> print (second_fifty_result[0].json)
# ...
#
#     # To get the whole response json from the MOST RECENT response
#     # (which will hold 50 individual responses depending on limit set):
# >>> print (search_service.most_recent_json)

# id;"screen_name";"description";
# "created_at";"location";"is_verified";
# "latest_following_count";"latest_followers_count";
# "latest_status_count";"array_agg"
# None
