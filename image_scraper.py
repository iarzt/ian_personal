from bs4 import BeautifulSoup
import requests
import re
import urllib.request

def get_image_links(url):
    response = requests.request('GET', url)
    soup = BeautifulSoup(response.text.encode('utf8'), 'html.parser')
    bigger_soup = soup.find_all("script", attrs = {'type':"application/ld+json"})
    
    bigger_soup = str(bigger_soup)
    
    image_links = re.findall('http:a//texassports.com/images/2019/9/20/[a-zA-Z]+_[a-zA-Z]+_2019.jpg', bigger_soup)
    
    return image_links

def get_name(image_link):
    
    name = image_link.replace('http://texassports.com/images/2019/9/20/', '')
    name = name.replace('https://texassports.com/images/2019/9/20/', '')
    name = name.replace('_2019.jpg', '')
    name = name.split('_')
    name.reverse()
    return '_'.join(name)

def save_image(image_link, name):
    urllib.request.urlretrieve(image_link, name + ".jpg")
    
    
def main():
    url = 'https://texassports.com/sports/baseball/roster'
    links = get_image_links(url)
    for link in links:
        name = get_name(link)
        save_image(link, name)
main()
