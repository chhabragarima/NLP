import sys
import math
import nltk
import re
import string
import os
import spacy
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from re import search
from nltk.tag import StanfordNERTagger
from fuzzywuzzy import fuzz
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

ennlp = spacy.load('en')
wd = WordNetLemmatizer()
stanford_classifier = os.environ.get('STANFORD_MODELS').split(":")[0]
stanford_ner_path = os.environ.get('CLASSPATH').split(":")[0]
st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')

stop = stopwords.words('english')
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2017-06-09/')

template = sys.argv[1]+".templates"
fout = open(template,"w")

ALPHA = 0.4
BETA = 0.6

incident_list = ["arson","attack","bombing","kidnapping","robbery"]
weapon_list = []
target_list = []
skip_list = []

#perp_indivial list for processing
list_of_keywords = []
with open("individual.txt") as perp:
	for keyword in perp:
		list_of_keywords.append(keyword.rstrip("\r\n"))

#creating victim list
def create_victim():
	victims = []
	with open("victim.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			victims.append(i)

	victims = list(set(victims))
	return victims


#creating target_list
def create_target():

	target = []
	with open("target.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			target.append(i)

	target = list(set(target))

	return target


#creating skip words for individuals
def create_skip_indv():

	skip = []
	with open("skip_words1.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			skip.append(i)

	skip = list(set(skip))

	return skip


#creating skip words for target
def create_skip():

	skip = []
	with open("skip_words2.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			skip.append(i)

	skip = list(set(skip))

	return skip

#extracting id and corpus separately
def extract_id(file_name):
	f1 = open(file_name)
	unique_id = []
	corpus = []
	for line in f1:
		temp = line.rstrip("\n")
		corpus.append(temp)

		match = search(r"DEV-MUC3-[0-9]{4}",line) or search("TST1-MUC3-[0-9]{4}",line) or search("TST2-MUC4-[0-9]{4}",line)
		
		if match:
			unique_id.append(match.group())
	
	return unique_id, corpus


#creating weapon list
def create_weapons():
	with open("weapon.txt") as fs:
		weapons = []

		for line in fs:
			word = line.strip()
			if word not in weapons:
				weapons.append(word)

	weapons = list(set(weapons))
	return weapons


#creating organisation list
def create_org():
	organisation_name = [] 
	with open("organisation.txt") as f2:
		for line in f2:
			on = line.strip()
			organisation_name.append(on)
	org_list = list(set(organisation_name))
	return org_list


#key words for names calculation
def individual_gen():
	general_indiv = []
	with open("names.txt","r") as f1:
		for i in f1:
			general_indiv.append(i.strip())
	return general_indiv


#machine learning features
def feature(answer_key,text_key):
	sample = []
	label = []

	for files in os.listdir(text_key):
		ansextension = "".join([files,'.anskey'])
		textpath = os.path.join(text_key,files)
		anspath = os.path.join(answer_key,ansextension)
		f1 = open(textpath)
		f2 = open(anspath)
		text = f1.read()
		sample.append(text)
		answer = f2.read()
		answer = answer.split('\n')

		for s in answer:
			slot = s.split(':')
			if slot[0] == 'INCIDENT':
				l = slot[-1].strip()
				label.append(l)
		f1.close()
		f2.close()
	return sample,label	


#input creation for classifer
def text_input(filename):
	corpus = []
	article = []
	label_key = []
	with open(filename,'r') as f1:
		p = re.compile(r'(?:D|T)(?:E|S)(?:V|T)[0-9]*\-MUC[0-9]\-[0-9]{4}')
		for sentence in f1.readlines():
			sent1 = sentence.split('\n')
			p1 = p.match(sent1[0])
			if p.match(sent1[0]) is None:
				article.append(sentence)
			else:
				corpus.append("".join(article))
				label_key.append(p1.group())
				article = [sentence]
	corpus.append("".join(article))
	return label_key,corpus


#SVM classifier for incident
def ml_incident():
	global_dict = {}
	answer_key = 'developset/answers'
	text_key = 'developset/texts'

	raw_feat = feature(answer_key,text_key)
	x_train,y_train = raw_feat[0],raw_feat[-1]
	countVec = TfidfVectorizer(stop_words='english')
	x_trainCV = countVec.fit_transform(x_train)
	T = x_trainCV.toarray()
	id_values, corpus=text_input(sys.argv[1])
	x_testCV = countVec.transform(corpus[1:])
	L = x_testCV.toarray()
	svmtest = LinearSVC().fit(T, y_train)
	svm = svmtest.predict(L)
	global_dict = dict(zip(id_values, svm))
	return global_dict


#returning values from incident dictionary
def final_incident(val):
	final_incident = global_dict[val]
	return final_incident


#finding key word positions to break
def position(sentence):
	pos = -1
	for i in range(len(sentence)):
		if sentence[i] in 'of':
			pos = i
			break
		elif sentence[i] in 'against':
			pos = i
			break
	if pos != -1:
		return pos
	else:
		return -1

#searching relevant tag for pattern
def searching_pos(noun_p):
	kk = noun_p.split()
	tmp = nlp.pos_tag(kk[0])
	if(tmp[0][1] == "DT" or tmp[0][1] == "CD"):
		l = noun_p.lstrip(tmp[0][0])
		return str(l)
	else:
		return noun_p
	
#main victim slot
def victim(article):
	new_list = []

	name_p = function_names(article)
	name_q = set(name_p)
	for i in name_q: 
		for j in name_q:
			if i in j and i!=j:
				new_list.append(i)
	per_name = name_q - set(new_list)

	verblist = ['VB','VBD','VBN','VBP','VBG','VBZ','JJ']

	impwords = []
	p_sent = []
	select_sent = []
	sent1 = []
	sent2 = []
	sent3 = []

	sentence = sent_tokenize(article)
	for s in sentence:
		wordset = []
		wordtokens = word_tokenize(s)
		wordstok = filter(str.isalnum,wordtokens)

		for w in wordstok:
			postags = nlp.pos_tag(w)
			stemmed = w
			stemmed1 = w

			if(str(postags[0][1]) in verblist):	
				stemmed = wd.lemmatize(w,'v')
			if stemmed in victimlist:
				select_sent.append(s)
				impwords.append(stemmed)
				break
	for sw in sentence:
		words = word_tokenize(sw)
		gram = r'''
		OF: { "of" }
		G1: {<DT>*<NN|NN.><IN><DT>*<JJ>*<CD>*<NN|NN.>*<JJ>*<NN|NN.>+}
		G2: {<CD>* <NN|NN.>+<CC>*<NN|NN.>*<WP>*<VB.><RB|RB*>*<VB.>+}
		G3: {<VB.>+<DT>*<CD>*<JJ>*<NN|NN.>+<,>*<NN|NN.>*<,>*<CC>*<DT>*<NN|NN.>*}
		'''
		chunked_text = nltk.RegexpParser(gram)
		tokenised_words = word_tokenize(sw)
		pos_t = nlp.pos_tag(sw)
		a = []
		tree = chunked_text.parse(pos_t)

		for subtree in tree.subtrees(filter = lambda t: t.label()=='OF'):
			p_sent.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='G1'):
			sent1.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='G3'):
			sent2.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='G2'):
			sent3.append(" ".join([a for (a,b) in subtree.leaves()]))

	a = []
	short_listed = []
	string_s = " "

	if(sent2 != 0):
		np_pharse = []
		np_pharse11 = []

		for s in sent2:
			doc = ennlp(unicode(s))
			for np in doc.noun_chunks:
				np_pharse.append(np)
		np_pharse22 = list(set(np_pharse))

		for i in np_pharse22:
			if i:
				np_pharse11.append(i)

		for i in sent2:
			tw = word_tokenize(i)
			stw = wd.lemmatize(tw[0],'v')
			if stw in impwords:
				short_listed.append(i)	

		for i in short_listed:
			string_s = " "
			for j in per_name:
				if j.lower() in str(i):
					string_s = j.lower()
					a.append(string_s)
			if string_s == " ":			
				for np in np_pharse11:
					if str(np) in i:
						finale = searching_pos(str(np))
						if finale in general_indiv:
							a.append(finale)	
	short_sent = []
	b = []
	ss = " "
	if(sent1 != 0):
		np_pharse = []
		np_pharse11 = []

		for s in sent1:
			doc = ennlp(unicode(s))
			for np in doc.noun_chunks:
				np_pharse.append(np)
		np_pharse22 = list(set(np_pharse))

		for i in np_pharse22:
			if i:
				np_pharse11.append(i)

		for i in sent1:
			wt = word_tokenize(i)
			for w in wt:
				stemm = wd.lemmatize(w,'v')
				if stemm in victimlist:
					short_sent.append(i)
					break
		for i in short_sent:
			if "of" in i or 'against' in i or "at" in i:
				ss = " "
				for j in per_name:
					if j.lower() in str(i):
						ss = j
						b.append(ss)
						break
				if ss == " ":
					eachsent = str(i).split()
					pos = position(eachsent)
					if(pos != -1):
						pos_tagging = nlp.pos_tag(eachsent[pos+1])
						if pos_tagging[0][1] not in ['CD','DT']:	
							for k in general_indiv:
								if k in " ".join(eachsent[pos+1:]):
									b.append(" ".join(eachsent[pos+1:]))
						else:
							pos_tagging = nlp.pos_tag(eachsent[pos+2])
							if pos_tagging[0][1] not in ['CD','DT']:
								for k in general_indiv:
									if k in " ".join(eachsent[pos+2:]):
										b.append(" ".join(eachsent[pos+2:]))
							else:
								for k in general_indiv:
									if k in " ".join(eachsent[pos+3:]):
										b.append(" ".join(eachsent[pos+3:]))	
							
	short_sent_pp = []; c = []
	if(sent3 != 0):
		for i in sent3:
			wt = word_tokenize(i)
			stemm = wd.lemmatize(wt[-1],'v')
			if stemm in victimlist:
				short_sent_pp.append(i)
		
		for i in short_sent_pp:
			ss2 = " "
			for j in per_name:
				if j.lower() in str(i):
					ss2 = j
					c.append(ss2)
					break
			if ss2 == " ":
				doc=ennlp(unicode(i))
				passive_toks=[tok for tok in doc  if (tok.dep_ == "nsubjpass")]
				if passive_toks != []:
					for p in passive_toks:
						if str(p) in general_indiv:
							c.append(str(p))
	fin = []
	for i in a:
		fin.append(i)
	for i in b:
		fin.append(i)
	for i in c:
		fin.append(i)	

	if fin != []:
		fin = list(set(fin))
		return fin
	else:
		return -1	


#extracting organisation from the key words
def extract_org(article):
	sentence = sent_tokenize(article)
	noun_words = ['NN','NNPS','NNP','NNS','JJ']
	ll2 = " "
	for i in sentence:
		w = word_tokenize(i)
		p = nltk.pos_tag(w)
		ll1 = " "
		for k,v in p:
			if v in noun_words:
				k1 = string.capwords(k)
				ll1 = ll1 + " " + k1
			else:
				ll1 = ll1 + " " + k
		ll2 = ll2 + ll1

	s = st.tag(ll2.split())

	list_org = []
	named_entity = chucks_cal(s)
	named_entity_string = [" ".join([token for token, tag in ne]) for ne in named_entity]
	named_entity_string_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entity]

	for k,v in named_entity_string_tag:
		if v == "ORGANIZATION":
			list_org.append(k)
	return list_org
	
#organisation slot
def organisation(article):
	key = ['claim','members of','front','movement','actions against','responsible for']
	org_names = extract_org(article)
	sentence = sent_tokenize(article)
	org_value = "-"
	for s in sentence:
		for k in key:
			if k in s:
				for f in org_list:
					if f in s:
						org_value = f
	return org_value


#finding postion to break
def finding_pos(w, split_np):
	temp = 0
	for i in range(len(split_np)):
		if split_np[i] == w:
			temp = i
	return temp

#weapon slot
def weapons(article):
	np_pharse = []
	sentence = sent_tokenize(article)
	tag = []
	count = 0
	weapon_ans = "-"
	ans_list = []

	for s in sentence:
		doc = ennlp(unicode(s))
		for np in doc.noun_chunks:
			np_pharse.append(np)
		words = word_tokenize(str(s))

		for w in words:
			for wp in np_pharse:
				if w in weapon_list and w in str(wp):
					noun_chunk = str(wp)
					split_np = noun_chunk.split()
					pos = finding_pos(w,split_np)
					if(pos != 0):
						postag = nlp.pos_tag(split_np[pos-1])
						if postag[0][1] == 'JJ':
							weapon_ans = split_np[pos-1] + " " + w
							ans_list.append(weapon_ans)
							ans_list.append(w)
						else:
							weapon_ans = w
							ans_list.append(weapon_ans)
					else:
						weapon_ans = w
						ans_list.append(weapon_ans)
						
	ans_list = list(set(ans_list))
	if ans_list:
		return ans_list
	else:
		return -1


#getting the continous words
def chucks_cal(tagged_sent):
    cc_chunk = []
    cur_chunk = []

    for token, tag in tagged_sent:
        if tag != "O":
            cur_chunk.append((token, tag))
        else:
            if cur_chunk:
                cc_chunk.append(cur_chunk)
                cur_chunk = []

    if cur_chunk:
        cc_chunk.append(cur_chunk)
    return cc_chunk


#getting persion names
def function_names(article):
	s1 = article.lower()
	s = sent_tokenize(s1)
	noun_words = ['NN','NNPS','NNP','NNS','JJ']
	ll2 = " "
	for i in s:
		w = word_tokenize(i)
		p = nltk.pos_tag(w)
		ll1 = " "
		for k,v in p:
			if v in noun_words:
				k1 = string.capwords(k)
				ll1 = ll1 + " " + k1
			else:
				ll1 = ll1 + " " + k
		ll2 = ll2 + ll1

	s = st.tag(ll2.split())
	list_person = []

	named_entity = chucks_cal(s)
	named_entity_string = [" ".join([token for token, tag in ne]) for ne in named_entity]
	named_entity_string_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entity]

	for k,v in named_entity_string_tag:
		if v == "PERSON":
			list_person.append(str(k))
	return list_person


#target slot
def target(article):

	sent_extract = []
	sentence = sent_tokenize(article)

	required = ['NN1','NN2','NN3','VB1','VB2','VB3','VB4','SUB','NN4']

	gram = []
	grammer1 = r'''
	NN1: {<VB.|VB|NN.|NN>+<DT>*<IN>*<DT>*<NN|NN.>+}
	'''
	grammer2 = r'''
	NN2: {<NNS>+<IN><JJ>*<NN|NNS>*<CC>*<JJ>*<NN|NNS>+}
	'''
	grammer3 = r'''
	NN3: {<NN>*<IN>*<TO>*<CD>*<NN|NN.>+}
	'''
	grammer4 = r'''
	NN4: {<DT>*<JJ>*<NN>+<IN>+<DT>*<NN>+<VB.>*<NN>+}
	'''
	grammer5 = r'''
	VB1: {<VBG><NNS>}
	'''
	grammer6 = r'''
	VB2: {<NN>*<VB.><IN>*<DT><NN>+}
	'''
	grammer7 = r'''
	VB3: {<CD>*<NNS><VB.><IN>*<DT>*<NN>+}
	'''
	grammer8 = r'''
	VB4: {<NN>*<VBG><IN><NN><TO><NN><VB.>+}
	'''

	gram.append(grammer1)
	gram.append(grammer2)
	gram.append(grammer3)
	gram.append(grammer4)
	gram.append(grammer5)
	gram.append(grammer6)
	gram.append(grammer7)
	gram.append(grammer8)

	for ws in sentence:
		words = word_tokenize(ws)
		
		for g in gram:
			after_chunk = nltk.RegexpParser(g)
			pos_words = nlp.pos_tag(ws)
			tree = after_chunk.parse(pos_words)

			for subtree in tree.subtrees(filter = lambda t: t.label() in required):
				sent_extract.append(" ".join([a for (a,b) in subtree.leaves()]))
		


	short_sent = []
	target_words = []
	if(len(sent_extract) != 0):
		for i in sent_extract:
			wt = word_tokenize(i)

			for w in wt:
				stemm = wd.lemmatize(w,'v')

				if stemm in target_list:
					short_sent.append(i)
					target_words.append(w)
					break
		
		ss = []
		prev = " "
		select = " "

		tags = ['JJ','NN','NNS']
		
		start = end = 0	

		for i in short_sent:
			wws = word_tokenize(i)

			for w in range(len(wws)):

				if end >= w:
					continue

				if wws[w] in target_words: # skip if its a key word
					continue
				elif wws[w] in target_list: #skip some weapons
					continue
				elif wws[w] in skip_list: #skip locations and unnecessary adjectives
					continue
				
				cur = nlp.pos_tag(wws[w])
				cur_tag = cur[0][1]

				if cur_tag in tags:
					start = w
					tmp = w

					while (tmp < len(wws)) and ((nlp.pos_tag(wws[tmp]))[0][1] in ['NNS','NN']) and (wws[tmp] not in skip_list) and (wws[tmp] not in target_list) and (wws[tmp] not in target_words):
						tmp = tmp + 1

					end = tmp 
				
					if start == end:
						ss.append(wws[start])
					else:
						ss.append(" ".join(wws[i] for i in range(start, end)))
							
						
					#break


		if (len(ss) != 0):
			ss = list(set(ss))
			return ss
		else:
			return -1
	else:
		return -1


#perp individual slot
def perp_indv(article):
	sentence_tokens=sent_tokenize(article)

	set_of_perp_sentences=set()

	for single_sentence in sentence_tokens:
	
		for phrase in list_of_keywords:
			ratio=fuzz.token_set_ratio(single_sentence,phrase)
			if ratio>90:
				x=nlp.pos_tag(single_sentence.lower())
				set_of_perp_sentences.add(single_sentence)


	list_of_perp_indv=[]
	list_of_sentences=[]
	for single_sentence in set_of_perp_sentences:

			
				x=nlp.pos_tag(single_sentence.lower())

				gram= r'''
				G1: {<RB>*<NN|NN.>*<IN>*<JJ>*<NN|NN.>+<VB.>+<DT>*<VB.>*<CD>*<IN>*<DT>*<JJ>*<NN|NN.>+}
				G2: {<CD>*<IN>*<DT>*<NN|NN.>+<WP><VB.>+<IN><VB.>*<IN>*<DT>*<NN|NN.>+}
				G3: {<NN|NN.>*<VB.>+<IN>+<CD>*<JJ>*<NN|NN.>+}
				G4: {<CD>*<NN|NN.>+<VB.>+<JJ>*<DT>*<NN|NN.>*<IN>+<NN|NN.>+}
				G5: {<CD>*<IN><DT>*<NN|NN.>+<VB.>+<NN|NN.>+}
				G6: {<JJ>*<NN|NN.>+<VB.>+<RP>*<DT>*<NN|NN.>+<IN>*}
				
				G8: {<NN|NN.>+<VB.>+<DT>*<NN|NN.>+<IN>+<DT>*<NN|NN.>+}
				G9: {<CD>*<RB>*<NN|NN.>+<VB.>+<JJ>*<IN>+}
							'''

				chunked_text = nltk.RegexpParser(gram)
				tree = chunked_text.parse(x)

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G1'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))
			
	
				for subtree in tree.subtrees(filter = lambda t: t.label()=='G2'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G3'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G4'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G5'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G6'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G7'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G8'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G9'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G10'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))
	

	np_pharse=[]
	np_pharse11=[]
	np_pharse22=[]
	for s in set(list_of_sentences):
		doc = ennlp(unicode(s))

		for np in doc.noun_chunks:
			np_pharse.append(np)
			np_pharse22 = list(set(np_pharse))
		for i in np_pharse22:
			if i:
				np_pharse11.append(i)
	
	updated_nounphrases=[]
	for np in set(np_pharse11):
		x=nlp.pos_tag(str(np))
		flag=1		
		for i in range(0,len(x)):
			if x[i][0].encode('utf-8') in indv_list:
				flag=0
				break
		if flag==1:
			updated_nounphrases.append(np)
		
		
	
	finalans=[]
	for i in set(updated_nounphrases):	
		if(str(i)!=" "):
				if(len(finalans)!=0):
					if "the" in str(i):
						i=str(i).replace("the ","")
						finalans.append(str(i))
					else:
						finalans.append(str(i))
				else:
					if "the" in str(i):
						i=str(i).replace("the ","")
						finalans.append(str(i))
					else:
						finalans.append(str(i))
	if (len(finalans)!=0):
		return list(set(finalans))
	else:
		return -1


#main tempelate function
def features(article, article_id):

	fout.write("ID:             {}\n".format(article_id))

	incident = final_incident(article_id)
	fout.write("INCIDENT:       {}\n".format(incident.upper()))

	weapon = weapons(article)
	if weapon == -1:
		fout.write("WEAPON:         {}\n".format("-"))
	else:
		for i in range(len(weapon)):
			if i == 0:
				fout.write("WEAPON:         {}\n".format(weapon[i].upper()))
			else:
				fout.write("                {}\n".format(weapon[i].upper()))

	perp = perp_indv(article)
	if perp == -1:
		fout.write("PERP INDIV:     {}\n".format("-"))
	else:
		for i in range(len(perp)):
			if i == 0:
				fout.write("PERP INDIV:     {}\n".format(perp[i].upper()))
			else:
				fout.write("                {}\n".format(perp[i].upper()))


	org = organisation(article)
	fout.write("PERP ORG:       {}\n".format(org.upper()))

	tar = target(article)
	if tar == -1:
		fout.write("TARGET:         {}\n".format("-"))
	else:
		for i in range(len(tar)):
			if i == 0:
				fout.write("TARGET:         {}\n".format(tar[i].upper()))
			else:
				fout.write("                {}\n".format(tar[i].upper()))

	vict = victim(article)
	if vict == -1:
		fout.write("VICTIM:         {}\n".format("-"))
	else:
		for i in range(len(vict)):
			if i == 0:
				fout.write("VICTIM:         {}\n".format(vict[i].upper()))
			else:
				fout.write("                {}\n".format(vict[i].upper()))

	fout.write("\n")

#splitting the articles
def extract_article(unique_id, corpus):
	for i in range(len(corpus)):
		match = search(r"DEV-MUC3-[0-9]{4}",corpus[i]) or search(r"TST1-MUC3-[0-9]{4}",corpus[i]) or search(r"TST2-MUC4-[0-9]{4}",corpus[i])

		article = " "

		if match:
			for j in range(i+1, len(corpus)):
				idMatch = search(r"DEV-MUC3-[0-9]{4}",corpus[j]) or search("TST1-MUC3-[0-9]{4}",corpus[j]) or search("TST2-MUC4-[0-9]{4}",corpus[j])
				
				if idMatch:
					if idMatch.group() in unique_id:
						break
				else:
					article = article + "".join(corpus[j].lower()) + " "

			features(article, match.group())



#creating the neccessary list
unique_id, corpus = extract_id(sys.argv[1])
weapon_list = create_weapons()
target_list = create_target()
victimlist = create_victim()
indv_list = create_skip_indv()
skip_list = create_skip()
general_indiv = individual_gen()
global_dict=ml_incident()
org_list = create_org()
extract_article(unique_id, corpus)
fout.close()
