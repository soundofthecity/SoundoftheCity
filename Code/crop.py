from pydub import AudioSegment	
import os
import magic

for subdirs, dirs,files in os.walk(os.getcwd()):
	for file in files:
		try:
			path = os.path.join(subdirs,file)	
			sound = AudioSegment.from_file(path,"wav")
			part = sound[:4000]
			ex_path = subdirs+"/4s_/"
			if not os.path.exists(os.path.dirname(ex_path)):
				os.makedirs(os.path.dirname(ex_path))
			part.export(ex_path+path.split('/')[-1],format='wav')
		except Exception:
			pass
