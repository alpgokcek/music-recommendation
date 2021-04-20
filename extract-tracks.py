import json

'''
You have to provide the Spotify playlist to the system.
Steps:
- Get the ID of your playlist from Spotify.
- Get auth token without any permissions from https://developer.spotify.com/console/get-playlist
- Send a cURL request to URL below and replace the fields below in the misc/playlists directory.
curl -X "GET" "https://api.spotify.com/v1/playlists/{PLAYLIST_URL}" -H "Accept: application/json" -H "Content-Type: application/json" -H "Authorization: Bearer {BEARER_TOKEN}" > {OUTPUT_FILE_NAME}.json
- Update the PATH variable below.
'''

PATH = "./misc/playlists/this-is-billie-eilish.json"
playlist = None
with open(PATH, 'r') as f:
    playlist = json.load(f)

tracks = playlist['tracks']['items']

ids = {"trackIds": [], "trackNames": []}

for track in tracks:
    ids['trackIds'].append(track['track']['id'])
    ids['trackNames'].append(track['track']['name'])

with open('misc/out/{}.json'.format(playlist['name'].replace(' ', '-').replace('.', '').replace('|', '').replace(':', '').lower()), 'w+') as out:
    json.dump(ids, out)
