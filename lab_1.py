import re

# Zadanie 1

text1a = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
result1a = re.sub(r'\d', '', text1a)
print(result1a)

text1b = '<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>'
result1b = re.sub(r'(<([^>]+)>.*?)', '', text1b)
print(result1b)

text1c = 'Lorem ipsum dolor sit amet, consectetur;' \
         ' adipiscing elit. Sed eget mattis sem. Mauris egestas erat quam,' \
         ' ut faucibus eros congue et. In blandit, mi eu porta; lobortis,' \
         ' tortor nisl facilisis leo, at tristique augue risus eu risus.'
result1c = re.sub(r'[,;\.]', '', text1c)
print(result1c)

# Zadanie 2
# a - wyciagniecie tylko hashtagow

text2a = 'Lorem ipsum dolor sit amet,' \
        ' consectetur adipiscing elit. Sed #texting eget mattis sem.' \
        ' Mauris #frasista egestas erat #tweetext quam, ut faucibus eros' \
        ' #frasier congue et. In blandit, mi eu porta lobortis,' \
        ' tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.'
result2a= re.findall('#', text2a)
print(result2a)


# b - hashtag z tekstem
text2b = 'Lorem ipsum dolor sit amet,' \
        ' consectetur adipiscing elit. Sed #texting eget mattis sem.' \
        ' Mauris #frasista egestas erat #tweetext quam, ut faucibus eros' \
        ' #frasier congue et. In blandit, mi eu porta lobortis,' \
        ' tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.'
result2b= re.findall('#[a-z]+', text2b)
print(result2b)

# Zadanie 3

text3 = 'Lorem ipsum dolor :) sit amet, consectetur;' \
        ' adipiscing elit. Sed eget mattis sem. ;) Mauris ;(' \
        ' egestas erat quam, :< ut faucibus eros congue :> et.' \
        ' In blandit, mi eu porta;lobortis, tortor :-)' \
        ' nisl facilisis leo, at ;< tristique augue risus eu risus ;-).'
result3 = re.findall('[:;][)(<>]|[:;][-][)(<>]', text3)
print(result3)

# Zadanie 4 - screen na branchu - wszystkie zadanie zostały wykonane, screen przedstawia ostatnie
