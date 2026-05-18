# FairytaleQA Sentence Splitter Inspection

**Split:** train  
**Limit:** 30  
**Samples where count changed:** 12 / 30  
**Samples unchanged:** 18 / 30  
**Total old sentences:** 357  
**Total new sentences:** 336  

**Seed:** 42  
**Filter:** implicit (`ex_or_im == 'implicit'`)  

## Sample 1

**story_name:** the-boy-who-set-a-snare-for-the-sun  
**question:** how will the boy feel when he comes home unsuccessful ?  
**answer:** sad .  

**old sentence count:** 12  
**new sentence count:** 12  

### Old Split

[S0] at the time when the animals reigned in the earth , they had killed all the people but a girl and her little brother .
[S1] these two were living in fear , in an out - of - the - way place .
[S2] the boy was a perfect little pigmy , and never grew beyond the size of a mere infant .
[S3] the girl increased with her years , so that the task of providing food and shelter fell wholly upon her .
[S4] she went out daily to get wood for the lodge - fire , and she took her little brother with her that no mishap might befall him .
[S5] he was too little to leave alone .
[S6] a big bird , of a mischievous disposition , might have flown away with him .
[S7] she made him a bow and arrows , and said to him one day , " my little brother , i will leave you behind where i have been gathering the wood .
[S8] you must hide yourself , and you will soon see the snow - birds come and pick the worms out of the logs which i have piled up .
[S9] shoot one of them and bring it home .
[S10] " he obeyed her , and tried his best to kill one , but he came home unsuccessful .
[S11] his sister told him that he must not despair , but try again the next day .

### New Split

[S0] at the time when the animals reigned in the earth , they had killed all the people but a girl and her little brother .
[S1] these two were living in fear , in an out - of - the - way place .
[S2] the boy was a perfect little pigmy , and never grew beyond the size of a mere infant .
[S3] the girl increased with her years , so that the task of providing food and shelter fell wholly upon her .
[S4] she went out daily to get wood for the lodge - fire , and she took her little brother with her that no mishap might befall him .
[S5] he was too little to leave alone .
[S6] a big bird , of a mischievous disposition , might have flown away with him .
[S7] she made him a bow and arrows , and said to him one day , " my little brother , i will leave you behind where i have been gathering the wood . "
[S8] " you must hide yourself , and you will soon see the snow - birds come and pick the worms out of the logs which i have piled up . "
[S9] " shoot one of them and bring it home . "
[S10] he obeyed her , and tried his best to kill one , but he came home unsuccessful .
[S11] his sister told him that he must not despair , but try again the next day .

## Sample 2

**story_name:** the-brownie-of-the-lake  
**question:** why did the people make fun of barbaik ?  
**answer:** because her hose had no tail .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] she had hardly spoken when the horse appeared , and mounting on his back she started for the village where the wedding was to be held .
[S1] at first she was so delighted with the chance of a holiday from the work which she hated , that she noticed nothing , but very soon it struck her as odd that as she passed along the roads full of people they all laughed as they looked at her horse .
[S2] at length she caught some words uttered by one man to another .
[S3] ' why , the farmer 's wife has sold her horse 's tail !
[S4] ' and turned in her saddle .
[S5] yes ; it was true .
[S6] her horse had no tail !
[S7] she had forgotten to ask for one , and the wicked dwarfs had carried out her orders to the letter !

### New Split

[S0] she had hardly spoken when the horse appeared , and mounting on his back she started for the village where the wedding was to be held .
[S1] at first she was so delighted with the chance of a holiday from the work which she hated , that she noticed nothing , but very soon it struck her as odd that as she passed along the roads full of people they all laughed as they looked at her horse .
[S2] at length she caught some words uttered by one man to another .
[S3] ' why , the farmer 's wife has sold her horse 's tail ! '
[S4] and turned in her saddle .
[S5] yes ; it was true .
[S6] her horse had no tail !
[S7] she had forgotten to ask for one , and the wicked dwarfs had carried out her orders to the letter !

## Sample 3

**story_name:** the-little-spirit-or-boy-man  
**question:** how did the boy use the old moccasin ?  
**answer:** to trick the fish into eating it .  

**old sentence count:** 12  
**new sentence count:** 12  

### Old Split

[S0] the great fish said to the boy - man under water .
[S1] " what is that floating ?
[S2] " to which the boy - man replied : " go , take hold of it , swallow it as fast as you can .
[S3] it is a great delicacy .
[S4] " the fish darted toward the old shoe and swallowed it , making of it a mere mouthful .
[S5] the boy - man laughed in himself , but said nothing , till the fish was fairly caught , when he took hold of the line and began to pull himself in his fish - carriage ashore .
[S6] the sister , who was watching all this time , opened wide her eyes as the huge fish came up and up upon the shore .
[S7] she opened them still more when the fish seemed to speak .
[S8] she heard from within a voice , saying , " make haste and release me from this nasty place .
[S9] " it was her brother 's voice , which she was accustomed to obey .
[S10] she made haste with her knife to open a door in the side of the fish , from which the boy - man presently leaped forth .
[S11] he lost no time in ordering her to cut it up and dry it , telling her that their spring supply of meat was now provided .

### New Split

[S0] the great fish said to the boy - man under water .
[S1] " what is that floating ? "
[S2] to which the boy - man replied : " go , take hold of it , swallow it as fast as you can . "
[S3] " it is a great delicacy . "
[S4] the fish darted toward the old shoe and swallowed it , making of it a mere mouthful .
[S5] the boy - man laughed in himself , but said nothing , till the fish was fairly caught , when he took hold of the line and began to pull himself in his fish - carriage ashore .
[S6] the sister , who was watching all this time , opened wide her eyes as the huge fish came up and up upon the shore .
[S7] she opened them still more when the fish seemed to speak .
[S8] she heard from within a voice , saying , " make haste and release me from this nasty place . "
[S9] it was her brother 's voice , which she was accustomed to obey .
[S10] she made haste with her knife to open a door in the side of the fish , from which the boy - man presently leaped forth .
[S11] he lost no time in ordering her to cut it up and dry it , telling her that their spring supply of meat was now provided .

## Sample 4

**story_name:** the-three-crowns  
**question:** how did the king feel about the eldest prince after small stones fell on him ?  
**answer:** mad .  

**old sentence count:** 12  
**new sentence count:** 10  

### Old Split

[S0] when they came into the palace yard , the king himself opened the carriage door , for respect to his new son - in - law .
[S1] as soon as he turned the handle , a shower of small stones fell on his powdered wig and his silk coat , and down he fell under them .
[S2] there was great fright and some laughter , and the king , after he wiped the blood from his forehead , looked very cross at the eldest prince .
[S3] ' my lord , ' says he , ' i 'm very sorry for this accident , but i 'm not to blame .
[S4] i saw the young smith get into the carriage , and we never stopped a minute since .
[S5] ' ' it 's uncivil you were to him .
[S6] go , ' says he to the other prince , ' and bring the young smith here , and be polite .
[S7] ' ' never fear , ' says he .
[S8] but there 's some people that could n't be good - natured if they tried , and not a bit civiller was the new messenger than the old , and when the king opened the carriage door a second time , it 's shower of mud that came down on him .
[S9] ' there 's no use , ' says he , ' going on this way .
[S10] the fox never got a better messenger than himself .
[S11] '

### New Split

[S0] when they came into the palace yard , the king himself opened the carriage door , for respect to his new son - in - law .
[S1] as soon as he turned the handle , a shower of small stones fell on his powdered wig and his silk coat , and down he fell under them .
[S2] there was great fright and some laughter , and the king , after he wiped the blood from his forehead , looked very cross at the eldest prince .
[S3] ' my lord , ' says he , ' i 'm very sorry for this accident , but i 'm not to blame . '
[S4] ' i saw the young smith get into the carriage , and we never stopped a minute since . '
[S5] ' it 's uncivil you were to him . go , ' says he to the other prince , ' and bring the young smith here , and be polite . '
[S6] ' never fear , ' says he .
[S7] but there 's some people that could n't be good - natured if they tried , and not a bit civiller was the new messenger than the old , and when the king opened the carriage door a second time , it 's shower of mud that came down on him .
[S8] ' there 's no use , ' says he , ' going on this way . '
[S9] ' the fox never got a better messenger than himself . '

## Sample 5

**story_name:** ogre-of-rashomon  
**question:** how did watanabe feel to see the ogre ?  
**answer:** surprised .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] watanabe 's eyes grew large with wonder , for he saw that the ogre was taller than the great gate , his eyes were flashing like mirrors in the sunlight , and his huge mouth was wide open , and as the monster breathed , flames of fire shot out of his mouth .
[S1] the ogre thought to terrify his foe , but watanabe never flinched .
[S2] he attacked the ogre with all his strength , and thus they fought face to face for a long time .
[S3] at last the ogre , finding that he could neither frighten nor beat watanabe and that he might himself be beaten , took to flight .
[S4] but watanabe , determined not to let the monster escape , put spurs to his horse and gave chase .

### New Split

[S0] watanabe 's eyes grew large with wonder , for he saw that the ogre was taller than the great gate , his eyes were flashing like mirrors in the sunlight , and his huge mouth was wide open , and as the monster breathed , flames of fire shot out of his mouth .
[S1] the ogre thought to terrify his foe , but watanabe never flinched .
[S2] he attacked the ogre with all his strength , and thus they fought face to face for a long time .
[S3] at last the ogre , finding that he could neither frighten nor beat watanabe and that he might himself be beaten , took to flight .
[S4] but watanabe , determined not to let the monster escape , put spurs to his horse and gave chase .

## Sample 6

**story_name:** quarrel-of-monkey-and-crab  
**question:** why did the monkey take all the best persimmons for himself ?  
**answer:** he was greedy .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] he consented to go with the crab to pick the fruit for him .
[S1] when they both reached the spot , the monkey was astonished to see what a fine tree had sprung from the seed , and with what a number of ripe persimmons the branches were loaded .
[S2] he quickly climbed the tree and began to pluck and eat , as fast as he could , one persimmon after another .
[S3] each time he chose the best and ripest he could find , and went on eating till he could eat no more .
[S4] not one would he give to the poor hungry crab waiting below , and when he had finished there was little but the hard , unripe fruit left .

### New Split

[S0] he consented to go with the crab to pick the fruit for him .
[S1] when they both reached the spot , the monkey was astonished to see what a fine tree had sprung from the seed , and with what a number of ripe persimmons the branches were loaded .
[S2] he quickly climbed the tree and began to pluck and eat , as fast as he could , one persimmon after another .
[S3] each time he chose the best and ripest he could find , and went on eating till he could eat no more .
[S4] not one would he give to the poor hungry crab waiting below , and when he had finished there was little but the hard , unripe fruit left .

## Sample 7

**story_name:** the-adventures-of-gilla-na-chreck-an-gour  
**question:** why did the princess not smile when she was in the gallery ?  
**answer:** nothing amused her .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] at last tom came to one of the city gates , and the guards laughed and cursed at him instead of letting him through .
[S1] tom stood it all for a little time , but at last one of them -- out of fun , as he said -- drove his bagnet half an inch or so into his side .
[S2] tom did nothing but take the fellow by the scruff of his neck and the waistband of his corduroys and fling him into the canal .
[S3] some ran to pull the fellow out , and others to let manners into the vulgarian with their swords and daggers .
[S4] but a tap from his club sent them headlong into the moat or down on the stones , and they were soon begging him to stay his hands .
[S5] so at last one of them was glad enough to show tom the way to the palace yard .
[S6] there was the king and the queen , and the princess in a gallery , looking at all sorts of wrestling and sword - playing , and rinka - fadhas ( long dances ) and mumming , all to please the princess .
[S7] but not a smile came over her handsome face .
[S8] well , they all stopped when they seen the young giant , with his boy 's face and long black hair , and his short curly beard -- for his poor mother could n't afford to buy razhurs -- and his great strong arms and bare legs , and no covering but the goat - skin that reached from his waist to his knees .
[S9] but an envious wizened basthard of a fellow , with a red head , that wished to be married to the princess , and did n't like how she opened her eyes at tom , came forward , and asked his business very snappishly .

### New Split

[S0] at last tom came to one of the city gates , and the guards laughed and cursed at him instead of letting him through .
[S1] tom stood it all for a little time , but at last one of them -- out of fun , as he said -- drove his bagnet half an inch or so into his side .
[S2] tom did nothing but take the fellow by the scruff of his neck and the waistband of his corduroys and fling him into the canal .
[S3] some ran to pull the fellow out , and others to let manners into the vulgarian with their swords and daggers .
[S4] but a tap from his club sent them headlong into the moat or down on the stones , and they were soon begging him to stay his hands .
[S5] so at last one of them was glad enough to show tom the way to the palace yard .
[S6] there was the king and the queen , and the princess in a gallery , looking at all sorts of wrestling and sword - playing , and rinka - fadhas ( long dances ) and mumming , all to please the princess .
[S7] but not a smile came over her handsome face .
[S8] well , they all stopped when they seen the young giant , with his boy 's face and long black hair , and his short curly beard -- for his poor mother could n't afford to buy razhurs -- and his great strong arms and bare legs , and no covering but the goat - skin that reached from his waist to his knees .
[S9] but an envious wizened basthard of a fellow , with a red head , that wished to be married to the princess , and did n't like how she opened her eyes at tom , came forward , and asked his business very snappishly .

## Sample 8

**story_name:** youth-who-wanted-to-win-daughter-of-mother-in-corner  
**question:** why was the mother so pleased with the linen ?  
**answer:** she could use it to make clothes .  

**old sentence count:** 11  
**new sentence count:** 10  

### Old Split

[S0] when he stood before the door at home , he turned around ; and there lay many , many hundred yards of the finest linen , finer than the most skillful weaver could have spun .
[S1] " mother , come out , come out !
[S2] " called and cried the youth .
[S3] his mother came darting out , and asked what was the matter .
[S4] and when she saw the linen , stretching as far as she could see , and then a bit , she could not believe her eyes , until the youth told her how it all happened .
[S5] but when she had heard that , and had tested the linen between her fingers , she was so pleased that she , too , began to sing and dance .
[S6] then she took the linen , cut it , and sewed shirts from it for her son and herself .
[S7] the remainder she took to town and sold for a good price .
[S8] then for a time they lived in all joy and comfort .
[S9] but when that was over the woman had not a bite to eat in the house .
[S10] she told her son that it was the highest time for him to take service , and really do something , or else both of them would have to starve to death .

### New Split

[S0] when he stood before the door at home , he turned around ; and there lay many , many hundred yards of the finest linen , finer than the most skillful weaver could have spun .
[S1] " mother , come out , come out ! " called and cried the youth .
[S2] his mother came darting out , and asked what was the matter .
[S3] and when she saw the linen , stretching as far as she could see , and then a bit , she could not believe her eyes , until the youth told her how it all happened .
[S4] but when she had heard that , and had tested the linen between her fingers , she was so pleased that she , too , began to sing and dance .
[S5] then she took the linen , cut it , and sewed shirts from it for her son and herself .
[S6] the remainder she took to town and sold for a good price .
[S7] then for a time they lived in all joy and comfort .
[S8] but when that was over the woman had not a bite to eat in the house .
[S9] she told her son that it was the highest time for him to take service , and really do something , or else both of them would have to starve to death .

## Sample 9

**story_name:** quarrel-of-monkey-and-crab  
**question:** how did the crab feel when he found the rice-dumpling ?  
**answer:** excited .  

**old sentence count:** 6  
**new sentence count:** 5  

### Old Split

[S0] long , long ago , one bright autumn day in japan , it happened , that a pink - faced monkey and a yellow crab were playing together along the bank of a river .
[S1] as they were running about , the crab found a rice - dumpling and the monkey a persimmon - seed .
[S2] the crab picked up the rice - dumpling and showed it to the monkey , saying : " look what a nice thing i have found !
[S3] " then the monkey held up his persimmon - seed and said : " i also have found something good !
[S4] look !
[S5] "

### New Split

[S0] long , long ago , one bright autumn day in japan , it happened , that a pink - faced monkey and a yellow crab were playing together along the bank of a river .
[S1] as they were running about , the crab found a rice - dumpling and the monkey a persimmon - seed .
[S2] the crab picked up the rice - dumpling and showed it to the monkey , saying : " look what a nice thing i have found ! "
[S3] then the monkey held up his persimmon - seed and said : " i also have found something good ! "
[S4] " look ! "

## Sample 10

**story_name:** festivities-at-the-house-of-conan  
**question:** what will happen if fionn does not drink water from the other fountain ?  
**answer:** fionn will die .  

**old sentence count:** 14  
**new sentence count:** 14  

### Old Split

[S0] there was also in the house a ram with a white belly , a jet - black head , dark - green horns , and green feet ; and there was in the end of the house a hag covered with a dark ash - coloured garment .
[S1] there were no persons in the house except these .
[S2] the man at the door - post welcomed us ; and we five , having our five hounds with us , sat on the floor of the bruighean .
[S3] ' let submissive homage be done to fionn mac cumhaill and his people , ' said the man at the door - post .
[S4] ' my case is that of a man begging a request , but obtaining neither the smaller nor the greater part of it , ' said the giant .
[S5] nevertheless , he rose up and did respectful homage to us .
[S6] after a while i became suddenly thirsty , and no person present perceived it but caoilte , who began to complain bitterly on that account .
[S7] ' you have no cause to complain , caoilte , ' said the man of the door - post , ' but only to step outside and fetch a drink for fionn from whichever of the fountains you please .
[S8] ' caoilte did so , and fetched the bronze vessel brimful to me and gave me to drink .
[S9] i took a drink from it , and the water tasted like honey while i was drinking , but bitter as gall when i put the vessel from my lips ; so that darting pains and symptoms of death seized me and agonising pangs from the poisonous draught .
[S10] i could be but with difficulty recognised ; and the lamentation of caoilte on account of my being in that condition was greater than that he had before given vent to on account of my thirst .
[S11] the man at the door - post desired caoilte to go out and bring me a drink from the other fountain .
[S12] caoilte obeyed , and brought me the iron vessel brimful .
[S13] i never underwent so much hardship in battle or conflict as i then suffered while drinking , in consequence of the bitterness of the draught ; but as soon as i put the vessel from my lips i recovered my own colour and appearance , and that gave joy and happiness to my people .

### New Split

[S0] there was also in the house a ram with a white belly , a jet - black head , dark - green horns , and green feet ; and there was in the end of the house a hag covered with a dark ash - coloured garment .
[S1] there were no persons in the house except these .
[S2] the man at the door - post welcomed us ; and we five , having our five hounds with us , sat on the floor of the bruighean .
[S3] ' let submissive homage be done to fionn mac cumhaill and his people , ' said the man at the door - post .
[S4] ' my case is that of a man begging a request , but obtaining neither the smaller nor the greater part of it , ' said the giant .
[S5] nevertheless , he rose up and did respectful homage to us .
[S6] after a while i became suddenly thirsty , and no person present perceived it but caoilte , who began to complain bitterly on that account .
[S7] ' you have no cause to complain , caoilte , ' said the man of the door - post , ' but only to step outside and fetch a drink for fionn from whichever of the fountains you please . '
[S8] caoilte did so , and fetched the bronze vessel brimful to me and gave me to drink .
[S9] i took a drink from it , and the water tasted like honey while i was drinking , but bitter as gall when i put the vessel from my lips ; so that darting pains and symptoms of death seized me and agonising pangs from the poisonous draught .
[S10] i could be but with difficulty recognised ; and the lamentation of caoilte on account of my being in that condition was greater than that he had before given vent to on account of my thirst .
[S11] the man at the door - post desired caoilte to go out and bring me a drink from the other fountain .
[S12] caoilte obeyed , and brought me the iron vessel brimful .
[S13] i never underwent so much hardship in battle or conflict as i then suffered while drinking , in consequence of the bitterness of the draught ; but as soon as i put the vessel from my lips i recovered my own colour and appearance , and that gave joy and happiness to my people .

## Sample 11

**story_name:** shinansha-or-south-pointing-carriage  
**question:** why did the dense fog leave the royal army confused ?  
**answer:** the army could not see .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] kotei in time grew to manhood and succeeded his father the emperor yuhi .
[S1] his early reign was greatly troubled by the rebel shiyu .
[S2] this rebel wanted to make himself king , and many were the battles which he fought to this end .
[S3] shiyu was a wicked magician , his head was made of iron , and there was no man that could conquer him .
[S4] at last kotei declared war against the rebel and led his army to battle , and the two armies met on a plain called takuroku .
[S5] the emperor boldly attacked the enemy , but the magician brought down a dense fog upon the battlefield .
[S6] while the royal army were wandering about in confusion , trying to find their way , shiyu retreated with his troops , laughing at having fooled the royal army .

### New Split

[S0] kotei in time grew to manhood and succeeded his father the emperor yuhi .
[S1] his early reign was greatly troubled by the rebel shiyu .
[S2] this rebel wanted to make himself king , and many were the battles which he fought to this end .
[S3] shiyu was a wicked magician , his head was made of iron , and there was no man that could conquer him .
[S4] at last kotei declared war against the rebel and led his army to battle , and the two armies met on a plain called takuroku .
[S5] the emperor boldly attacked the enemy , but the magician brought down a dense fog upon the battlefield .
[S6] while the royal army were wandering about in confusion , trying to find their way , shiyu retreated with his troops , laughing at having fooled the royal army .

## Sample 12

**story_name:** mirror-of-matsuyama  
**question:** what did the father do because he thought the daughter was cursing the step-mother ?  
**answer:** scolded her .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] the girl was frightened by her father 's severity .
[S1] never had he spoken to her in such a tone .
[S2] her confusion changed to apprehension , her color from scarlet to white .
[S3] she sat dumb and shamefaced , unable to reply .
[S4] appearances were certainly against her ; the young girl looked guilty , and the father thinking that perhaps after all what his wife had told him was true , spoke angrily : " then , is it really true that you are daily cursing your step - mother and praying for her death ?
[S5] have you forgotten what i told you , that although she is your step - mother you must be obedient and loyal to her ?
[S6] what evil spirit has taken possession of your heart that you should be so wicked ?
[S7] you have certainly changed , my daughter !
[S8] what has made you so disobedient and unfaithful ?
[S9] " and the father 's eyes filled with sudden tears to think that he should have to upbraid his daughter in this way .

### New Split

[S0] the girl was frightened by her father 's severity .
[S1] never had he spoken to her in such a tone .
[S2] her confusion changed to apprehension , her color from scarlet to white .
[S3] she sat dumb and shamefaced , unable to reply .
[S4] appearances were certainly against her ; the young girl looked guilty , and the father thinking that perhaps after all what his wife had told him was true , spoke angrily : " then , is it really true that you are daily cursing your step - mother and praying for her death ? "
[S5] " have you forgotten what i told you , that although she is your step - mother you must be obedient and loyal to her ? "
[S6] " what evil spirit has taken possession of your heart that you should be so wicked ? "
[S7] " you have certainly changed , my daughter ! "
[S8] " what has made you so disobedient and unfaithful ? "
[S9] and the father 's eyes filled with sudden tears to think that he should have to upbraid his daughter in this way .

## Sample 13

**story_name:** pinkel-thief  
**question:** how did the two eldest brothers feel about pinkel's employment under the king ?  
**answer:** dislike .  

**old sentence count:** 30  
**new sentence count:** 29  

### Old Split

[S0] long , long ago there lived a widow who had three sons .
[S1] the two eldest were grown up , and though they were known to be idle fellows , some of the neighbours had given them work to do on account of the respect in which their mother was held .
[S2] but at the time this story begins they had both been so careless and idle that their masters declared they would keep them no longer .
[S3] so home they went to their mother and youngest brother , of whom they thought little , because he made himself useful about the house , and looked after the hens , and milked the cow .
[S4] ' pinkel , ' they called him in scorn , and by - and - by ' pinkel ' became his name throughout the village .
[S5] the way was further than they thought , and it was morning before they reached the palace .
[S6] now , at last , their luck seemed to have turned , for while the two eldest were given places in the king 's stables , pinkel was taken as page to the little prince .
[S7] he was a clever and amusing boy , who saw everything that passed under his eyes , and the king noticed this , and often employed him in his own service , which made his brothers very jealous .
[S8] things went on this way for some time , and pinkel every day rose in the royal favour .
[S9] at length the envy of his brothers became so great that they could bear it no longer , and consulted together how best they might ruin his credit with the king .
[S10] they did not wish to kill him - though , perhaps , they would not have been sorry if they had heard he was dead - but merely wished to remind him that he was after all only a child , not half so old and wise as they .
[S11] it may easily be guessed that all this made the brothers more envious than they were before ; and they cast about in their minds afresh how best they might destroy him .
[S12] at length they remembered the goat with golden horns and the bells , and they rejoiced ; ' for , ' said they , ' this time the old woman will be on the watch , and let him be as clever as he likes , the bells on the horns are sure to warn her .
[S13] ' so when , as before , the king came down to the stables and praised the cleverness of their brother , the young men told him of that other marvel possessed by the witch , the goat with the golden horns .
[S14] from this moment the king never closed his eyes at night for longing after this wonderful creature .
[S15] he understood something of the danger that there might be in trying to steal it , now that the witch 's suspicions were aroused , and he spent hours in making plans for outwitting her .
[S16] but somehow he never could think of anything that would do , and at last , as the brothers had foreseen , he sent for pinkel .
[S17] as soon as he had reached the middle of the lake , pinkel took the wool out of the bells , which began to tinkle loudly .
[S18] their sound awoke the witch , who cried out as before : ' is that you , pinkel ?
[S19] ' ' yes , dear mother , it is i , ' said pinkel .
[S20] ' have you stolen my golden goat ?
[S21] ' asked she .
[S22] ' yes , dear mother , i have , ' answered pinkel .
[S23] ' are you not a knave , pinkel ?
[S24] ' ' yes , dear mother , i am , ' he replied .
[S25] and the old witch shouted in a rage : ' ah !
[S26] beware how you come hither again , for next time you shall not escape me !
[S27] ' but pinkel laughed and rowed on .
[S28] the king was so delighted with the goat that he always kept it by his side , night and day ; and , as he had promised , pinkel was made ruler over the third part of the kingdom .
[S29] as may be supposed , the brothers were more furious than ever , and grew quite thin with rage .

### New Split

[S0] long , long ago there lived a widow who had three sons .
[S1] the two eldest were grown up , and though they were known to be idle fellows , some of the neighbours had given them work to do on account of the respect in which their mother was held .
[S2] but at the time this story begins they had both been so careless and idle that their masters declared they would keep them no longer .
[S3] so home they went to their mother and youngest brother , of whom they thought little , because he made himself useful about the house , and looked after the hens , and milked the cow .
[S4] ' pinkel , ' they called him in scorn , and by - and - by ' pinkel ' became his name throughout the village .
[S5] the way was further than they thought , and it was morning before they reached the palace .
[S6] now , at last , their luck seemed to have turned , for while the two eldest were given places in the king 's stables , pinkel was taken as page to the little prince .
[S7] he was a clever and amusing boy , who saw everything that passed under his eyes , and the king noticed this , and often employed him in his own service , which made his brothers very jealous .
[S8] things went on this way for some time , and pinkel every day rose in the royal favour .
[S9] at length the envy of his brothers became so great that they could bear it no longer , and consulted together how best they might ruin his credit with the king .
[S10] they did not wish to kill him - though , perhaps , they would not have been sorry if they had heard he was dead - but merely wished to remind him that he was after all only a child , not half so old and wise as they .
[S11] it may easily be guessed that all this made the brothers more envious than they were before ; and they cast about in their minds afresh how best they might destroy him .
[S12] at length they remembered the goat with golden horns and the bells , and they rejoiced ; ' for , ' said they , ' this time the old woman will be on the watch , and let him be as clever as he likes , the bells on the horns are sure to warn her . '
[S13] so when , as before , the king came down to the stables and praised the cleverness of their brother , the young men told him of that other marvel possessed by the witch , the goat with the golden horns .
[S14] from this moment the king never closed his eyes at night for longing after this wonderful creature .
[S15] he understood something of the danger that there might be in trying to steal it , now that the witch 's suspicions were aroused , and he spent hours in making plans for outwitting her .
[S16] but somehow he never could think of anything that would do , and at last , as the brothers had foreseen , he sent for pinkel .
[S17] as soon as he had reached the middle of the lake , pinkel took the wool out of the bells , which began to tinkle loudly .
[S18] their sound awoke the witch , who cried out as before : ' is that you , pinkel ? '
[S19] ' yes , dear mother , it is i , ' said pinkel .
[S20] ' have you stolen my golden goat ? ' asked she .
[S21] ' yes , dear mother , i have , ' answered pinkel .
[S22] ' are you not a knave , pinkel ? '
[S23] ' yes , dear mother , i am , ' he replied .
[S24] and the old witch shouted in a rage : ' ah ! '
[S25] ' beware how you come hither again , for next time you shall not escape me ! '
[S26] but pinkel laughed and rowed on .
[S27] the king was so delighted with the goat that he always kept it by his side , night and day ; and , as he had promised , pinkel was made ruler over the third part of the kingdom .
[S28] as may be supposed , the brothers were more furious than ever , and grew quite thin with rage .

## Sample 14

**story_name:** the-raspberry-worm  
**question:** how were the girls able to get home ?  
**answer:** they followed the bird .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] ' oh , please do n't do that , ' cried both the girls , very frightened .
[S1] ' well , for your sake i will forgive him , ' said the old man , ' i am not revengeful .
[S2] greetings to otto and tell him that he may expect a gift from me , too .
[S3] good - bye .
[S4] ' the two girls , light of heart , now took their berries and ran off through the wood after the bird ; and soon it began to get lighter in the wood and they wondered how they could have lost their way yesterday , it seemed so easy and plain now .

### New Split

[S0] ' oh , please do n't do that , ' cried both the girls , very frightened .
[S1] ' well , for your sake i will forgive him , ' said the old man , ' i am not revengeful . '
[S2] ' greetings to otto and tell him that he may expect a gift from me , too . '
[S3] ' good - bye . '
[S4] the two girls , light of heart , now took their berries and ran off through the wood after the bird ; and soon it began to get lighter in the wood and they wondered how they could have lost their way yesterday , it seemed so easy and plain now .

## Sample 15

**story_name:** youth-who-was-to-serve-three-years-without-pay  
**question:** how did the man feel about the kitten ?  
**answer:** letdown .  

**old sentence count:** 23  
**new sentence count:** 18  

### Old Split

[S0] the following morning it was no better .
[S1] the man set out as early as possible , and had not as yet reached town before he met the old woman with the basket .
[S2] " good - day , granny , " said the man .
[S3] " and good - day to you , daddy , " said the old woman .
[S4] " what have you in your basket to - day ?
[S5] " asked the man .
[S6] " if you want to know , then buy it !
[S7] " was again the answer .
[S8] " what does it cost ?
[S9] " asked the man .
[S10] she wanted four shillings for it , she had only the one price .
[S11] the man said he would buy it , for he thought that this time he would make a better purchase .
[S12] he raised the cover , and this time a kitten lay in the basket .
[S13] when he reached home , there stood the youth , waiting to see what he was to get in lieu of his second year 's wages .
[S14] " are you back again , master !
[S15] " said he .
[S16] " yes , indeed , " said the master .
[S17] " what did you buy to - day ?
[S18] " asked the youth .
[S19] " alas , nothing better than i did yesterday , " said the man , " but i did as we agreed , and bought the first thing i came across , and that was this kitten .
[S20] " " you could not have hit on anything better , " said the youth , " for all my life long i have been fond of cats as well as of dogs .
[S21] " " i do not fare so badly this way , " thought the man , " but when he sets out for himself , then the matter will probably turn out differently .
[S22] "

### New Split

[S0] the following morning it was no better .
[S1] the man set out as early as possible , and had not as yet reached town before he met the old woman with the basket .
[S2] " good - day , granny , " said the man .
[S3] " and good - day to you , daddy , " said the old woman .
[S4] " what have you in your basket to - day ? " asked the man .
[S5] " if you want to know , then buy it ! "
[S6] was again the answer .
[S7] " what does it cost ? " asked the man .
[S8] she wanted four shillings for it , she had only the one price .
[S9] the man said he would buy it , for he thought that this time he would make a better purchase .
[S10] he raised the cover , and this time a kitten lay in the basket .
[S11] when he reached home , there stood the youth , waiting to see what he was to get in lieu of his second year 's wages .
[S12] " are you back again , master ! " said he .
[S13] " yes , indeed , " said the master .
[S14] " what did you buy to - day ? " asked the youth .
[S15] " alas , nothing better than i did yesterday , " said the man , " but i did as we agreed , and bought the first thing i came across , and that was this kitten . "
[S16] " you could not have hit on anything better , " said the youth , " for all my life long i have been fond of cats as well as of dogs . "
[S17] " i do not fare so badly this way , " thought the man , " but when he sets out for himself , then the matter will probably turn out differently . "

## Sample 16

**story_name:** east-of-sun-and-west-of-moon  
**question:** why did the north wind blow himself up and make himself large and thick ?  
**answer:** the journey was long .  

**old sentence count:** 16  
**new sentence count:** 16  

### Old Split

[S0] " o , yes , i know very well where the castle lies , " said the north wind .
[S1] " i blew an aspen leaf there just once , and then i was so weary that i could not blow at all for many a long day .
[S2] but if you want to get there above all things , and are not afraid of me , i will take you on my back , and see whether i can blow you there .
[S3] " the maiden said that she must and would get to the castle , if it were by any means possible .
[S4] she was not afraid , no matter how hard the journey might be .
[S5] " very well , then you must stay here over night , " said the north wind .
[S6] " for if we are to get there to - morrow , we must have the whole day before us .
[S7] " early the next morning the north wind awakened the maiden .
[S8] then he blew himself up , and made himself so large and thick that he was quite horrible to look at .
[S9] thereupon they rushed along through the air as though they meant to reach the end of the world at once .
[S10] and everywhere beneath them raged such a storm that forests were pulled out by the roots , and houses torn down .
[S11] as they rushed across the sea , ships foundered by the hundreds .
[S12] further and further they went , so far that no one could even imagine it , and still they were flying across the sea .
[S13] gradually the north wind grew weary , and became weaker and weaker .
[S14] finally he could hardly keep going , and sank lower and lower .
[S15] at last he flew so low that the waves washed his ankles .

### New Split

[S0] " o , yes , i know very well where the castle lies , " said the north wind .
[S1] " i blew an aspen leaf there just once , and then i was so weary that i could not blow at all for many a long day . "
[S2] " but if you want to get there above all things , and are not afraid of me , i will take you on my back , and see whether i can blow you there . "
[S3] the maiden said that she must and would get to the castle , if it were by any means possible .
[S4] she was not afraid , no matter how hard the journey might be .
[S5] " very well , then you must stay here over night , " said the north wind .
[S6] " for if we are to get there to - morrow , we must have the whole day before us . "
[S7] early the next morning the north wind awakened the maiden .
[S8] then he blew himself up , and made himself so large and thick that he was quite horrible to look at .
[S9] thereupon they rushed along through the air as though they meant to reach the end of the world at once .
[S10] and everywhere beneath them raged such a storm that forests were pulled out by the roots , and houses torn down .
[S11] as they rushed across the sea , ships foundered by the hundreds .
[S12] further and further they went , so far that no one could even imagine it , and still they were flying across the sea .
[S13] gradually the north wind grew weary , and became weaker and weaker .
[S14] finally he could hardly keep going , and sank lower and lower .
[S15] at last he flew so low that the waves washed his ankles .

## Sample 17

**story_name:** jack-my-hedgehog  
**question:** why did the farmer know nothing about jack my hedgehog all this time ?  
**answer:** jack my hedgehog never went home .  

**old sentence count:** 6  
**new sentence count:** 6  

### Old Split

[S0] jack my hedgehog continued to herd his pigs , and they increased in number till there were so many that the forest seemed full of them .
[S1] so he made up his mind to live there no longer , and sent a message to his father telling him to have all the stables and outhouses in the village cleared , as he was going to bring such an enormous herd that all who would might kill what they chose .
[S2] his father was much vexed at this news , for he thought jack had died long ago .
[S3] jack my hedgehog mounted his cock , and driving his pigs before him into the village , he let every one kill as many as they chose , and such a hacking and hewing of pork went on as you might have heard for miles off .
[S4] then said jack , ' daddy , let the blacksmith shoe my cock once more ; then i 'll ride off , and i promise you i 'll never come back again as long as i live .
[S5] ' so the father had the cock shod , and rejoiced at the idea of getting rid of his son .

### New Split

[S0] jack my hedgehog continued to herd his pigs , and they increased in number till there were so many that the forest seemed full of them .
[S1] so he made up his mind to live there no longer , and sent a message to his father telling him to have all the stables and outhouses in the village cleared , as he was going to bring such an enormous herd that all who would might kill what they chose .
[S2] his father was much vexed at this news , for he thought jack had died long ago .
[S3] jack my hedgehog mounted his cock , and driving his pigs before him into the village , he let every one kill as many as they chose , and such a hacking and hewing of pork went on as you might have heard for miles off .
[S4] then said jack , ' daddy , let the blacksmith shoe my cock once more ; then i 'll ride off , and i promise you i 'll never come back again as long as i live . '
[S5] so the father had the cock shod , and rejoiced at the idea of getting rid of his son .

## Sample 18

**story_name:** momotaro-story-of-son-of-peach  
**question:** how did momotaro feel towards his parents ?  
**answer:** grateful .  

**old sentence count:** 4  
**new sentence count:** 3  

### Old Split

[S0] one day momotaro came to his foster - father and said solemnly : " father , by a strange chance we have become father and son .
[S1] your goodness to me has been higher than the mountain grasses which it was your daily work to cut , and deeper than the river where my mother washes the clothes .
[S2] i do not know how to thank you enough .
[S3] "

### New Split

[S0] one day momotaro came to his foster - father and said solemnly : " father , by a strange chance we have become father and son . "
[S1] " your goodness to me has been higher than the mountain grasses which it was your daily work to cut , and deeper than the river where my mother washes the clothes . "
[S2] " i do not know how to thank you enough . "

## Sample 19

**story_name:** jamie-freel-and-the-young-lady  
**question:** what will jamie do to seek his fortune ?  
**answer:** look for the wee folk .  

**old sentence count:** 17  
**new sentence count:** 14  

### Old Split

[S0] these articles of attire had long been ready for a certain triste ceremony , in which she would some day fill the chief part , and only saw the light occasionally when they were hung out to air .
[S1] she was willing to give even these to the fair trembling visitor , who was turning in dumb sorrow and wonder from her to jamie , and from jamie back to her .
[S2] the poor girl suffered herself to be dressed , and then sat down on a " creepie " in the chimney corner and buried her face in her hands .
[S3] " what 'll we do to keep up a lady like you ?
[S4] " cried the old woman .
[S5] " i 'll work for you both , mother , " replied the son .
[S6] " an ' how could a lady live on we'er poor diet ?
[S7] " she repeated .
[S8] " i 'll work for her , " was all jamie 's answer .
[S9] he kept his word .
[S10] the young lady was very sad for a long time , and tears stole down her cheeks many an evening , while the old woman span by the fire and jamie made salmon nets , an accomplishment acquired by him in hopes of adding to the comfort of their guest .
[S11] but she was always gentle , and tried to smile when she perceived them looking at her .
[S12] by degrees she adapted herself to their ways and mode of life .
[S13] it was not very long before she began to feed the pig , mash potatoes and meal for the fowls , and knit blue worsted socks .
[S14] so a year passed and halloween came round again .
[S15] " mother , " said jamie , taking down his cap , " i 'm off to the old castle to seek my fortune .
[S16] "

### New Split

[S0] these articles of attire had long been ready for a certain triste ceremony , in which she would some day fill the chief part , and only saw the light occasionally when they were hung out to air .
[S1] she was willing to give even these to the fair trembling visitor , who was turning in dumb sorrow and wonder from her to jamie , and from jamie back to her .
[S2] the poor girl suffered herself to be dressed , and then sat down on a " creepie " in the chimney corner and buried her face in her hands .
[S3] " what 'll we do to keep up a lady like you ? " cried the old woman .
[S4] " i 'll work for you both , mother , " replied the son .
[S5] " an ' how could a lady live on we'er poor diet ? " she repeated .
[S6] " i 'll work for her , " was all jamie 's answer .
[S7] he kept his word .
[S8] the young lady was very sad for a long time , and tears stole down her cheeks many an evening , while the old woman span by the fire and jamie made salmon nets , an accomplishment acquired by him in hopes of adding to the comfort of their guest .
[S9] but she was always gentle , and tried to smile when she perceived them looking at her .
[S10] by degrees she adapted herself to their ways and mode of life .
[S11] it was not very long before she began to feed the pig , mash potatoes and meal for the fowls , and knit blue worsted socks .
[S12] so a year passed and halloween came round again .
[S13] " mother , " said jamie , taking down his cap , " i 'm off to the old castle to seek my fortune . "

## Sample 20

**story_name:** notscha  
**question:** how did notscha feel when he found his temple destroyed ?  
**answer:** sad .  

**old sentence count:** 13  
**new sentence count:** 13  

### Old Split

[S0] now notscha had been absent in the spirit upon that day .
[S1] when he returned he found his temple destroyed ; and the spirit of the hill gave him the details .
[S2] notscha hurried to his master and related with tears what had befallen him .
[S3] the latter was roused and said : " it is li dsing 's fault .
[S4] after you had given back your body to your parents , you were no further concern of his .
[S5] why should he withdraw from you the enjoyment of the incense ?
[S6] " then the great one made a body of lotus - plants , gave it the gift of life , and enclosed the soul of notscha within it .
[S7] this done he called out in a loud voice : " arise !
[S8] " a drawing of breath was heard , and notscha leaped up once more in the shape of a small boy .
[S9] he flung himself down before his master and thanked him .
[S10] the latter bestowed upon him the magic of the fiery lance , and notscha thenceforward had two whirling wheels beneath his feet : the wheel of the wind and the wheel of fire .
[S11] with these he could rise up and down in the air .
[S12] the master also gave him a bag of panther - skin in which to keep his armlet and his silken cloth .

### New Split

[S0] now notscha had been absent in the spirit upon that day .
[S1] when he returned he found his temple destroyed ; and the spirit of the hill gave him the details .
[S2] notscha hurried to his master and related with tears what had befallen him .
[S3] the latter was roused and said : " it is li dsing 's fault . "
[S4] " after you had given back your body to your parents , you were no further concern of his . "
[S5] " why should he withdraw from you the enjoyment of the incense ? "
[S6] then the great one made a body of lotus - plants , gave it the gift of life , and enclosed the soul of notscha within it .
[S7] this done he called out in a loud voice : " arise ! "
[S8] a drawing of breath was heard , and notscha leaped up once more in the shape of a small boy .
[S9] he flung himself down before his master and thanked him .
[S10] the latter bestowed upon him the magic of the fiery lance , and notscha thenceforward had two whirling wheels beneath his feet : the wheel of the wind and the wheel of fire .
[S11] with these he could rise up and down in the air .
[S12] the master also gave him a bag of panther - skin in which to keep his armlet and his silken cloth .

## Sample 21

**story_name:** the-three-crowns  
**question:** why did the second prince go down to the well ?  
**answer:** the eldest prince did not come back .  

**old sentence count:** 23  
**new sentence count:** 23  

### Old Split

[S0] well , they were n't crossing the lake while a cat would be lickin ' her ear , and the poor men could n't stir hand or foot to follow them .
[S1] they saw seven inches handing the three princesses out of the boat , and letting them down by a basket into a draw - well , but king nor princes ever saw an opening before in the same place .
[S2] when the last lady was out of sight , the men found the strength in their arms and legs again .
[S3] round the lake they ran , and never drew rein till they came to the well and windlass ; and there was the silk rope rolled on the axle , and the nice white basket hanging to it .
[S4] ' let me down , ' says the youngest prince .
[S5] ' i 'll die or recover them again .
[S6] ' ' no , ' says the second daughter 's sweetheart , ' it is my turn first .
[S7] ' and says the other , ' i am the eldest .
[S8] ' so they gave way to him , and in he got into the basket , and down they let him .
[S9] first they lost sight of him , and then , after winding off a hundred perches of the silk rope , it slackened , and they stopped turning .
[S10] they waited two hours , and then they went to dinner , because there was no pull made at the rope .
[S11] guards were set till next morning , and then down went the second prince , and sure enough , the youngest of all got himself let down on the third day .
[S12] he went down perches and perches , while it was as dark about him as if he was in a big pot with a cover on .
[S13] at last he saw a glimmer far down , and in a short time he felt the ground .
[S14] out he came from the big lime - kiln , and , lo !
[S15] and behold you , there was a wood , and green fields , and a castle in a lawn , and a bright sky over all .
[S16] ' it 's in tir - na - n - oge i am , ' says he .
[S17] ' let 's see what sort of people are in the castle .
[S18] ' on he walked , across fields and lawn , and no one was there to keep him out or let him into the castle ; but the big hall - door was wide open .
[S19] he went from one fine room to another that was finer , and at last he reached the handsomest of all , with a table in the middle .
[S20] and such a dinner as was laid upon it !
[S21] the prince was hungry enough , but he was too mannerly to eat without being invited .
[S22] so he sat by the fire , and he did not wait long till he heard steps , and in came seven inches with the youngest sister by the hand .

### New Split

[S0] well , they were n't crossing the lake while a cat would be lickin ' her ear , and the poor men could n't stir hand or foot to follow them .
[S1] they saw seven inches handing the three princesses out of the boat , and letting them down by a basket into a draw - well , but king nor princes ever saw an opening before in the same place .
[S2] when the last lady was out of sight , the men found the strength in their arms and legs again .
[S3] round the lake they ran , and never drew rein till they came to the well and windlass ; and there was the silk rope rolled on the axle , and the nice white basket hanging to it .
[S4] ' let me down , ' says the youngest prince .
[S5] ' i 'll die or recover them again . '
[S6] ' no , ' says the second daughter 's sweetheart , ' it is my turn first . '
[S7] and says the other , ' i am the eldest . '
[S8] so they gave way to him , and in he got into the basket , and down they let him .
[S9] first they lost sight of him , and then , after winding off a hundred perches of the silk rope , it slackened , and they stopped turning .
[S10] they waited two hours , and then they went to dinner , because there was no pull made at the rope .
[S11] guards were set till next morning , and then down went the second prince , and sure enough , the youngest of all got himself let down on the third day .
[S12] he went down perches and perches , while it was as dark about him as if he was in a big pot with a cover on .
[S13] at last he saw a glimmer far down , and in a short time he felt the ground .
[S14] out he came from the big lime - kiln , and , lo !
[S15] and behold you , there was a wood , and green fields , and a castle in a lawn , and a bright sky over all .
[S16] ' it 's in tir - na - n - oge i am , ' says he .
[S17] ' let 's see what sort of people are in the castle . '
[S18] on he walked , across fields and lawn , and no one was there to keep him out or let him into the castle ; but the big hall - door was wide open .
[S19] he went from one fine room to another that was finer , and at last he reached the handsomest of all , with a table in the middle .
[S20] and such a dinner as was laid upon it !
[S21] the prince was hungry enough , but he was too mannerly to eat without being invited .
[S22] so he sat by the fire , and he did not wait long till he heard steps , and in came seven inches with the youngest sister by the hand .

## Sample 22

**story_name:** the-iron-stove  
**question:** why did the iron stove tell the miller's daughter to go away at once and tell the king's daughter to come ?  
**answer:** she revealed that she was the miller 's daughter .  

**old sentence count:** 13  
**new sentence count:** 12  

### Old Split

[S0] then he gave her someone for a guide , who walked near her and said nothing , but he brought her in two hours to her house .
[S1] there was great joy in the castle when the princess came back , and the old king fell on her neck and kissed her .
[S2] but she was very much troubled , and said , ' dear father , listen to what has befallen me !
[S3] i should never have come home again out of the great wild wood if i had not come to an iron stove , to whom i have had to promise that i will go back to free him and marry him !
[S4] ' the old king was so frightened that he nearly fainted , for she was his only daughter .
[S5] so they consulted together , and determined that the miller 's daughter , who was very beautiful , should take her place .
[S6] they took her there , gave her a knife , and said she must scrape at the iron stove .
[S7] she scraped for twenty - four hours , but did not make the least impression .
[S8] when the day broke , a voice called from the iron stove , ' it seems to me that it is day outside .
[S9] ' then she answered , ' it seems so to me ; i think i hear my father 's mill rattling .
[S10] ' ' so you are a miller 's daughter !
[S11] then go away at once , and tell the king 's daughter to come .
[S12] '

### New Split

[S0] then he gave her someone for a guide , who walked near her and said nothing , but he brought her in two hours to her house .
[S1] there was great joy in the castle when the princess came back , and the old king fell on her neck and kissed her .
[S2] but she was very much troubled , and said , ' dear father , listen to what has befallen me ! '
[S3] ' i should never have come home again out of the great wild wood if i had not come to an iron stove , to whom i have had to promise that i will go back to free him and marry him ! '
[S4] the old king was so frightened that he nearly fainted , for she was his only daughter .
[S5] so they consulted together , and determined that the miller 's daughter , who was very beautiful , should take her place .
[S6] they took her there , gave her a knife , and said she must scrape at the iron stove .
[S7] she scraped for twenty - four hours , but did not make the least impression .
[S8] when the day broke , a voice called from the iron stove , ' it seems to me that it is day outside . '
[S9] then she answered , ' it seems so to me ; i think i hear my father 's mill rattling . '
[S10] ' so you are a miller 's daughter ! '
[S11] ' then go away at once , and tell the king 's daughter to come . '

## Sample 23

**story_name:** the-believing-husbands  
**question:** why wasn't the young woman in the kitchen or the dairy ?  
**answer:** she got the horses ' dinner .  

**old sentence count:** 11  
**new sentence count:** 10  

### Old Split

[S0] they worked hard for many hours .
[S1] at length grew hungry , so the young woman was sent home to bring them food , and also to give the horses their dinner .
[S2] when she went into the stables , she suddenly saw the heavy pack - saddle of the speckled mare just over her head .
[S3] she jumped and said to herself : ' suppose that pack - saddle were to fall and kill me , how dreadful it would be !
[S4] ' and she sat down just under the pack - saddle she was so much afraid of , and began to cry .
[S5] now the others out on the moor grew hungrier and hungrier .
[S6] ' what can have become of her ?
[S7] ' asked they .
[S8] at length the mother declared that she would wait no longer , and must go and see what had happened .
[S9] as the bride was nowhere in the kitchen or the dairy , the old woman went into the stable .
[S10] she found her daughter weeping bitterly .

### New Split

[S0] they worked hard for many hours .
[S1] at length grew hungry , so the young woman was sent home to bring them food , and also to give the horses their dinner .
[S2] when she went into the stables , she suddenly saw the heavy pack - saddle of the speckled mare just over her head .
[S3] she jumped and said to herself : ' suppose that pack - saddle were to fall and kill me , how dreadful it would be ! '
[S4] and she sat down just under the pack - saddle she was so much afraid of , and began to cry .
[S5] now the others out on the moor grew hungrier and hungrier .
[S6] ' what can have become of her ? ' asked they .
[S7] at length the mother declared that she would wait no longer , and must go and see what had happened .
[S8] as the bride was nowhere in the kitchen or the dairy , the old woman went into the stable .
[S9] she found her daughter weeping bitterly .

## Sample 24

**story_name:** king-kojata  
**question:** why did hyacinthia breathe onto the window ?  
**answer:** so she could escape with the prince .  

**old sentence count:** 28  
**new sentence count:** 27  

### Old Split

[S0] the prince returned to his room in despair ; then the princess hyacinthia came to him once more changed into the likeness of a bee , and asked him , ' why so sad , prince milan ?
[S1] ' ' how can i help being sad ?
[S2] your father has set me this time an impossible task .
[S3] before a candle which he has lit burns to the socket , i am to make a pair of boots .
[S4] but what does a prince know of shoemaking ?
[S5] if i ca n't do it , i lose my head .
[S6] ' ' and what do you mean to do ?
[S7] ' asked hyacinthia .
[S8] ' well , what is there to be done ?
[S9] what he demands i ca n't and wo n't do , so he must just make an end of me .
[S10] ' ' not so , dearest .
[S11] i love you dearly , and you shall marry me , and i 'll either save your life or die with you .
[S12] we must fly now as quickly as we can , for there is no other way of escape .
[S13] ' with these words she breathed on the window , and her breath froze on the pane .
[S14] then she led milan out of the room with her , shut the door , and threw the key away .
[S15] hand in hand , they hurried to the spot where they had descended into the lower world , and at last reached the banks of the lake .
[S16] prince milan 's charger was still grazing on the grass which grew near the water .
[S17] the horse no sooner recognized his master , than it neighed loudly with joy , and springing towards him , it stood as if rooted to the ground , while prince milan and hyacinthia jumped on its back .
[S18] then it sped onwards like an arrow from a bow .
[S19] in the meantime the magician was waiting impatiently for the prince .
[S20] enraged by the delay , he sent his servants to fetch him , for the appointed time was past .
[S21] the servants came to the door , and finding it locked , they knocked ; but the frozen breath on the window replied in prince milan 's voice , ' i am coming directly .
[S22] ' with this answer they returned to the magician .
[S23] but when the prince still did not appear , after a time he sent his servants a second time to bring him .
[S24] the frozen breath always gave the same answer , but the prince never came .
[S25] at last the magician lost all patience , and commanded the door to be burst open .
[S26] but when his servants did so , they found the room empty , and the frozen breath laughed aloud .
[S27] out of his mind with rage , the magician ordered the prince to be pursued .

### New Split

[S0] the prince returned to his room in despair ; then the princess hyacinthia came to him once more changed into the likeness of a bee , and asked him , ' why so sad , prince milan ? '
[S1] ' how can i help being sad ? '
[S2] ' your father has set me this time an impossible task . '
[S3] ' before a candle which he has lit burns to the socket , i am to make a pair of boots . '
[S4] ' but what does a prince know of shoemaking ? '
[S5] ' if i ca n't do it , i lose my head . '
[S6] ' and what do you mean to do ? ' asked hyacinthia .
[S7] ' well , what is there to be done ? '
[S8] ' what he demands i ca n't and wo n't do , so he must just make an end of me . '
[S9] ' not so , dearest . '
[S10] ' i love you dearly , and you shall marry me , and i 'll either save your life or die with you . '
[S11] ' we must fly now as quickly as we can , for there is no other way of escape . '
[S12] with these words she breathed on the window , and her breath froze on the pane .
[S13] then she led milan out of the room with her , shut the door , and threw the key away .
[S14] hand in hand , they hurried to the spot where they had descended into the lower world , and at last reached the banks of the lake .
[S15] prince milan 's charger was still grazing on the grass which grew near the water .
[S16] the horse no sooner recognized his master , than it neighed loudly with joy , and springing towards him , it stood as if rooted to the ground , while prince milan and hyacinthia jumped on its back .
[S17] then it sped onwards like an arrow from a bow .
[S18] in the meantime the magician was waiting impatiently for the prince .
[S19] enraged by the delay , he sent his servants to fetch him , for the appointed time was past .
[S20] the servants came to the door , and finding it locked , they knocked ; but the frozen breath on the window replied in prince milan 's voice , ' i am coming directly . '
[S21] with this answer they returned to the magician .
[S22] but when the prince still did not appear , after a time he sent his servants a second time to bring him .
[S23] the frozen breath always gave the same answer , but the prince never came .
[S24] at last the magician lost all patience , and commanded the door to be burst open .
[S25] but when his servants did so , they found the room empty , and the frozen breath laughed aloud .
[S26] out of his mind with rage , the magician ordered the prince to be pursued .

## Sample 25

**story_name:** jorinde-and-joringel  
**question:** how will joringel feel when he finds the flower ?  
**answer:** happy .  

**old sentence count:** 6  
**new sentence count:** 6  

### Old Split

[S0] he plucked this flower and went with it to the castle ; and there everything which he touched with the flower was freed from the enchantment , and he got his jorinde back again through it .
[S1] when he awoke in the morning he began to seek mountain and valley to find such a flower .
[S2] he sought it for eight days , and on the ninth early in the morning he found the blood - red flower .
[S3] in its centre was a large dew - drop , as big as the most lovely pearl .
[S4] he travelled day and night with this flower till he arrived at the castle .
[S5] when he came within a hundred paces of it he did not cease to be able to move , but he went on till he reached the gate .

### New Split

[S0] he plucked this flower and went with it to the castle ; and there everything which he touched with the flower was freed from the enchantment , and he got his jorinde back again through it .
[S1] when he awoke in the morning he began to seek mountain and valley to find such a flower .
[S2] he sought it for eight days , and on the ninth early in the morning he found the blood - red flower .
[S3] in its centre was a large dew - drop , as big as the most lovely pearl .
[S4] he travelled day and night with this flower till he arrived at the castle .
[S5] when he came within a hundred paces of it he did not cease to be able to move , but he went on till he reached the gate .

## Sample 26

**story_name:** jack-my-hedgehog  
**question:** how did the king feel when jack returned his daughter to him ?  
**answer:** shocked .  

**old sentence count:** 9  
**new sentence count:** 9  

### Old Split

[S0] then jack my hedgehog set off for the first kingdom , and there the king had given strict orders that if anyone should be seen riding a cock and carrying a bagpipe he was to be chased away and shot at , and on no account to be allowed to enter the palace .
[S1] so when jack my hedgehog rode up the guards charged him with their bayonets , but he put spurs to his cock , flew up over the gate right to the king 's windows , let himself down on the sill , and called out that if he was not given what had been promised him , both the king and his daughter should pay for it with their lives .
[S2] then the king coaxed and entreated his daughter to go with jack and so save both their lives .
[S3] the princess dressed herself all in white , and her father gave her a coach with six horses and servants in gorgeous liveries and quantities of money .
[S4] she stepped into the coach , and jack my hedgehog with his cock and pipes took his place beside her .
[S5] they both took leave , and the king fully expected never to set eyes on them again .
[S6] but matters turned out very differently from what he had expected , for when they had got a certain distance from the town jack tore all the princess 's smart clothes off her , and pricked her all over with his bristles , saying : ' that 's what you get for treachery .
[S7] now go back , i 'll have no more to say to you .
[S8] ' and with that he hunted her home , and she felt she had been disgraced and put to shame till her life 's end .

### New Split

[S0] then jack my hedgehog set off for the first kingdom , and there the king had given strict orders that if anyone should be seen riding a cock and carrying a bagpipe he was to be chased away and shot at , and on no account to be allowed to enter the palace .
[S1] so when jack my hedgehog rode up the guards charged him with their bayonets , but he put spurs to his cock , flew up over the gate right to the king 's windows , let himself down on the sill , and called out that if he was not given what had been promised him , both the king and his daughter should pay for it with their lives .
[S2] then the king coaxed and entreated his daughter to go with jack and so save both their lives .
[S3] the princess dressed herself all in white , and her father gave her a coach with six horses and servants in gorgeous liveries and quantities of money .
[S4] she stepped into the coach , and jack my hedgehog with his cock and pipes took his place beside her .
[S5] they both took leave , and the king fully expected never to set eyes on them again .
[S6] but matters turned out very differently from what he had expected , for when they had got a certain distance from the town jack tore all the princess 's smart clothes off her , and pricked her all over with his bristles , saying : ' that 's what you get for treachery . '
[S7] ' now go back , i 'll have no more to say to you . '
[S8] and with that he hunted her home , and she felt she had been disgraced and put to shame till her life 's end .

## Sample 27

**story_name:** the-enchanted-moccasins  
**question:** how will ko-ko feel about fighting the father ?  
**answer:** excited .  

**old sentence count:** 15  
**new sentence count:** 12  

### Old Split

[S0] " ho !
[S1] ho !
[S2] who is there ?
[S3] " cried the wicked father , making his appearance at the opening and looking down .
[S4] " it is i , onwee bahmondang !
[S5] " cried ko - koor , thinking to frighten the wicked father .
[S6] " ah !
[S7] it is you , is it ?
[S8] i will be there presently , " called the old man .
[S9] " do not be in haste to go away !
[S10] " ko - ko , observing that the old man was in earnest , scrambled up from the ground , and set off promptly at his highest rate of speed .
[S11] when he looked back and saw that the wicked father was gaining upon him , ko - koor mounted a tree , as had onwee bahmondang before , and fired off a number of arrows , but as they were no more than common arrows , he got nothing by it , but was obliged to descend , and run again for life .
[S12] as he hurried on he encountered the skeleton of a moose , into which he would have transformed himself , but not having the slightest confidence in any one of all the guardians who should have helped him , he passed on .
[S13] the wicked father was hot in pursuit , and ko - koor was suffering terribly for lack of wind , when luckily he remembered the enchanted moccasins .
[S14] he could not send them to the end of the earth , as had onwee bahmondang .

### New Split

[S0] " ho ! "
[S1] " ho ! "
[S2] " who is there ? " cried the wicked father , making his appearance at the opening and looking down .
[S3] " it is i , onwee bahmondang ! " cried ko - koor , thinking to frighten the wicked father .
[S4] " ah ! "
[S5] " it is you , is it ? i will be there presently , " called the old man .
[S6] " do not be in haste to go away ! "
[S7] ko - ko , observing that the old man was in earnest , scrambled up from the ground , and set off promptly at his highest rate of speed .
[S8] when he looked back and saw that the wicked father was gaining upon him , ko - koor mounted a tree , as had onwee bahmondang before , and fired off a number of arrows , but as they were no more than common arrows , he got nothing by it , but was obliged to descend , and run again for life .
[S9] as he hurried on he encountered the skeleton of a moose , into which he would have transformed himself , but not having the slightest confidence in any one of all the guardians who should have helped him , he passed on .
[S10] the wicked father was hot in pursuit , and ko - koor was suffering terribly for lack of wind , when luckily he remembered the enchanted moccasins .
[S11] he could not send them to the end of the earth , as had onwee bahmondang .

## Sample 28

**story_name:** youth-who-wanted-to-win-daughter-of-mother-in-corner  
**question:** how did the woman feel about her son singing and dancing ?  
**answer:** unhappy .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] once upon a time there was a woman who had a son .
[S1] he was so lazy and slow that there was not a single blessed useful thing he would do .
[S2] but he liked to sing and to dance , and that is what he did all day long , and far into the night as well .
[S3] the longer this went on , the worse off his mother was .
[S4] the youth was growing , and he wanted so much to eat that it was barely possible to find it .
[S5] more and more went for his clothes the older he grew , since his clothes did not last long , as you may imagine , because the youth skipped and dance about without stopping , through forest and field .
[S6] at length it was too much for his mother .
[S7] one day she told the young fellow that he ought at last to get to work , and really do something , or both of them would have to starve to death .
[S8] but the youth had no mind to do so , he said , and would rather try to win the daughter of the mother in the corner .
[S9] if he got her , then he would live happily ever after , and could sing and dance , and would not have to plague himself with work .

### New Split

[S0] once upon a time there was a woman who had a son .
[S1] he was so lazy and slow that there was not a single blessed useful thing he would do .
[S2] but he liked to sing and to dance , and that is what he did all day long , and far into the night as well .
[S3] the longer this went on , the worse off his mother was .
[S4] the youth was growing , and he wanted so much to eat that it was barely possible to find it .
[S5] more and more went for his clothes the older he grew , since his clothes did not last long , as you may imagine , because the youth skipped and dance about without stopping , through forest and field .
[S6] at length it was too much for his mother .
[S7] one day she told the young fellow that he ought at last to get to work , and really do something , or both of them would have to starve to death .
[S8] but the youth had no mind to do so , he said , and would rather try to win the daughter of the mother in the corner .
[S9] if he got her , then he would live happily ever after , and could sing and dance , and would not have to plague himself with work .

## Sample 29

**story_name:** the-brown-bear-of-norway  
**question:** how did the youngest princess feel when she saw the lady standing by them ?  
**answer:** scared .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] but all her care was in vain .
[S1] another evening , when they were all so happy , and the prince dandling the baby , a beautiful greyhound stood before them , took the child out of the father 's hand , and was out of the door before you could wink .
[S2] this time she shouted and ran out of the room , but there were some of the servants in the next room , and all declared that neither child nor dog passed out .
[S3] she felt , somehow , as if it was her husband 's fault , but still she kept command over herself , and did n't once reproach him .
[S4] when the third child was born she would hardly allow a window or a door to be left open for a moment ; but she was n't the nearer to keep the child to herself .
[S5] they were sitting one evening by the fire , when a lady appeared standing by them .
[S6] the princess opened her eyes in a great fright and stared at her , and while she was doing so , the lady wrapped a shawl round the baby that was sitting in its father 's lap , and either sank through the ground with it or went up through the wide chimney .
[S7] this time the mother kept her bed for a month .

### New Split

[S0] but all her care was in vain .
[S1] another evening , when they were all so happy , and the prince dandling the baby , a beautiful greyhound stood before them , took the child out of the father 's hand , and was out of the door before you could wink .
[S2] this time she shouted and ran out of the room , but there were some of the servants in the next room , and all declared that neither child nor dog passed out .
[S3] she felt , somehow , as if it was her husband 's fault , but still she kept command over herself , and did n't once reproach him .
[S4] when the third child was born she would hardly allow a window or a door to be left open for a moment ; but she was n't the nearer to keep the child to herself .
[S5] they were sitting one evening by the fire , when a lady appeared standing by them .
[S6] the princess opened her eyes in a great fright and stared at her , and while she was doing so , the lady wrapped a shawl round the baby that was sitting in its father 's lap , and either sank through the ground with it or went up through the wide chimney .
[S7] this time the mother kept her bed for a month .

## Sample 30

**story_name:** flax  
**question:** why does the fern sing mournfully ?  
**answer:** it 's knotted .  

**old sentence count:** 8  
**new sentence count:** 7  

### Old Split

[S0] " ah , yes , no doubt , " said the fern , " but you do not know the world yet as well as i do , for my sticks are knotty " ; and then it sang quite mournfully : " snip , snap , snurre , basse lurre .
[S1] the song is ended .
[S2] " " no , it is not ended , " said the flax .
[S3] " to - morrow the sun will shine or the rain descend .
[S4] i feel that i am growing .
[S5] i feel that i am in full blossom .
[S6] i am the happiest of all creatures , for i may some day come to something .
[S7] "

### New Split

[S0] " ah , yes , no doubt , " said the fern , " but you do not know the world yet as well as i do , for my sticks are knotty " ; and then it sang quite mournfully : " snip , snap , snurre , basse lurre . "
[S1] " the song is ended . "
[S2] " no , it is not ended , " said the flax .
[S3] " to - morrow the sun will shine or the rain descend . "
[S4] " i feel that i am growing . "
[S5] " i feel that i am in full blossom . "
[S6] " i am the happiest of all creatures , for i may some day come to something . "
