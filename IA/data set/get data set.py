import requests
import re
import os
from bs4 import BeautifulSoup

def GetTiny(artist):
    url = "https://genius.com/artists/" + str(artist)
    response = requests.get(url)

    if response.status_code == 200:
        content = response.content.decode('utf-8')
        urls = re.findall(r"<a[^>]*href\s*=\s*['\"]([^'\"]+)['\"][^>]*>", content)
        urls_albums = []

        for url in urls:
            if not url.startswith("https://"):
                url = "https://genius.com" + url
            if url.startswith("https://genius.com/albums/" + str(artist)):
                if url not in urls_albums:
                    urls_albums.append(url)

        path_fodler_artist = f"{os.path.dirname(os.path.abspath(__file__))}\Tiny {artist}"
        os.mkdir(path_fodler_artist)
        
        for url in urls_albums:
            print(" || "+url+" || ")
            album = url.split("/")[5]
            path_album_artist = f"{path_fodler_artist}\{album}"
            
            if not os.path.exists(path_album_artist):
                os.mkdir(path_album_artist)
            else:
                path_album_artist += "-bug"
                os.mkdir(path_album_artist)
                
            album_response = requests.get(url)
            
            if album_response.status_code == 200:
                album_content = album_response.content.decode('utf-8')
                album_urls = re.findall(r"<a[^>]*href\s*=\s*['\"]([^'\"]+)['\"][^>]*>", album_content)
                urls_song = []
                
                for url in album_urls:
                    if "https://genius.com/"+ str(artist) in url:
                        response_text = requests.get(url)

                        if response_text.status_code == 200:
                            
                            try:
                                song_name = str(url).replace("https://genius.com/" + str(artist), "").replace("-", " ").replace("lyrics", "").strip()
                                with open(f"{path_album_artist}\{song_name}.txt", "w", encoding="utf-8") as file:
                                    file.write("")
                                    
                                print(f"  | {song_name} | ")
                                    
                                content_text = response_text.content.decode('utf-8')
                                soup = BeautifulSoup(content_text, 'html.parser')
                                divs_with_class = soup.find_all('div', {'class': 'Lyrics__Container-sc-1ynbvzw-5'})
                                
                                for index, target_div in enumerate(divs_with_class, start=1):
                                    with open(f"{path_album_artist}\{song_name}.txt", "a", encoding="utf-8") as file:
                                        file.write(str(target_div.get_text(separator="\n")) + "\n")
                                    with open(f"{path_fodler_artist}\\all_data_artist.txt", "a", encoding="utf-8") as file:
                                        file.write(str(target_div.get_text(separator="\n")) + "\n")
                                    with open(f"{os.path.dirname(os.path.abspath(__file__))}\\all_artist_data_artist.txt", "a", encoding="utf-8") as file:
                                        file.write(str(target_div.get_text(separator="\n")) + "\n")
                            except:
                                pass
GetTiny("Ed-sheeran")
print(" || FINISH || ")