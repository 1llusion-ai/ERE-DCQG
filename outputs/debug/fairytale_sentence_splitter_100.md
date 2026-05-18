# FairytaleQA Sentence Splitter Inspection

**Split:** train  
**Limit:** 100  
**Samples where count changed:** 45 / 100  
**Samples unchanged:** 55 / 100  
**Total old sentences:** 1122  
**Total new sentences:** 1050  

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

## Sample 31

**story_name:** jack-my-hedgehog  
**question:** why did jack prick the princess ?  
**answer:** the king cheated on his promise .  

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

## Sample 32

**story_name:** shinansha-or-south-pointing-carriage  
**question:** why did kotei declare war against shiyu ?  
**answer:** shiyu was a rebel .  

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

## Sample 33

**story_name:** the-tale-of-benjamin-bunny  
**question:** what did little benjamin and peter do when they heard the noises ?  
**answer:** they hid themselves and the onions underneath a large basket .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] they got amongst flower - pots , and frames and tubs ; peter heard noises worse than ever , his eyes were as big as lolly - pops !
[S1] he was a step or two in front of his cousin , when he suddenly stopped .
[S2] this is what those little rabbits saw round that corner !
[S3] little benjamin took one look , and then , in half a minute less than no time , he hid himself and peter and the onions underneath a large basket ....
[S4] the cat got up and stretched herself , and came and sniffed at the basket .
[S5] perhaps she liked the smell of onions !
[S6] anyway , she sat down upon the top of the basket .
[S7] she sat there for five hours .

### New Split

[S0] they got amongst flower - pots , and frames and tubs ; peter heard noises worse than ever , his eyes were as big as lolly - pops !
[S1] he was a step or two in front of his cousin , when he suddenly stopped .
[S2] this is what those little rabbits saw round that corner !
[S3] little benjamin took one look , and then , in half a minute less than no time , he hid himself and peter and the onions underneath a large basket ....
[S4] the cat got up and stretched herself , and came and sniffed at the basket .
[S5] perhaps she liked the smell of onions !
[S6] anyway , she sat down upon the top of the basket .
[S7] she sat there for five hours .

## Sample 34

**story_name:** a-french-puck  
**question:** how did the wedding guests feel after they saw the dress ?  
**answer:** amazed .  

**old sentence count:** 17  
**new sentence count:** 15  

### Old Split

[S0] ' what a beautiful girl !
[S1] ' exclaimed the men .
[S2] ' what a lovely dress !
[S3] ' whispered the women .
[S4] but just as she entered the church and took the hand of the bridegroom , who was waiting for her , a loud noise was heard .
[S5] ' crick !
[S6] crack !
[S7] crick !
[S8] crack !
[S9] ' and the wedding garments fell to the ground , to the great confusion of the wearer .
[S10] not that the ceremony was put off for a little thing like that !
[S11] cloaks in profusion were instantly offered to the young bride , but she was so upset that she could hardly keep from tears .
[S12] one of the guests , more curious than the rest , stayed behind to examine the dress , determined , if she could , to find out the cause of the disaster .
[S13] ' the thread must have been rotten , ' she said to herself .
[S14] ' i will see if i can break it .
[S15] ' but search as she would she could find none .
[S16] the thread had vanished !

### New Split

[S0] ' what a beautiful girl ! ' exclaimed the men .
[S1] ' what a lovely dress ! ' whispered the women .
[S2] but just as she entered the church and took the hand of the bridegroom , who was waiting for her , a loud noise was heard .
[S3] ' crick ! '
[S4] ' crack ! '
[S5] ' crick ! '
[S6] ' crack ! '
[S7] and the wedding garments fell to the ground , to the great confusion of the wearer .
[S8] not that the ceremony was put off for a little thing like that !
[S9] cloaks in profusion were instantly offered to the young bride , but she was so upset that she could hardly keep from tears .
[S10] one of the guests , more curious than the rest , stayed behind to examine the dress , determined , if she could , to find out the cause of the disaster .
[S11] ' the thread must have been rotten , ' she said to herself .
[S12] ' i will see if i can break it . '
[S13] but search as she would she could find none .
[S14] the thread had vanished !

## Sample 35

**story_name:** the-toad-woman  
**question:** why did the toad-woman think her children insulted the young man ?  
**answer:** the children had fat in their mouths .  

**old sentence count:** 7  
**new sentence count:** 6  

### Old Split

[S0] she at once set forth ; and she was no sooner out of sight than the young man and his dog , spirit - iron , blowing a strong breath in the face of the toad - woman 's four children ( who were all bad spirits , or bear - fiends ) , they put out their life .
[S1] they then set them up by the side of the door , having first thrust a piece of the white fat in each of their mouths .
[S2] the toad - woman spent a long time in finding the bear which she had been sent after , and she made at least five and twenty attempts before she was able to climb to the carcass .
[S3] she slipped down three times where she went up once .
[S4] when she returned with the great bear on her back , as she drew near her lodge she was astonished to see the four children standing up by the door - posts with the fat in their mouths .
[S5] she was angry with them , and called out : " why do you thus insult the pomatum of your brother ?
[S6] "

### New Split

[S0] she at once set forth ; and she was no sooner out of sight than the young man and his dog , spirit - iron , blowing a strong breath in the face of the toad - woman 's four children ( who were all bad spirits , or bear - fiends ) , they put out their life .
[S1] they then set them up by the side of the door , having first thrust a piece of the white fat in each of their mouths .
[S2] the toad - woman spent a long time in finding the bear which she had been sent after , and she made at least five and twenty attempts before she was able to climb to the carcass .
[S3] she slipped down three times where she went up once .
[S4] when she returned with the great bear on her back , as she drew near her lodge she was astonished to see the four children standing up by the door - posts with the fat in their mouths .
[S5] she was angry with them , and called out : " why do you thus insult the pomatum of your brother ? "

## Sample 36

**story_name:** story-of-old-man-who-made-withered-trees-to-flower  
**question:** why was shiro barking and digging ?  
**answer:** there was something in the ground .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] one day shiro was heard barking for a long time in the field at the back of his master 's house .
[S1] the old man , thinking that perhaps some birds were attacking the corn , hurried out to see what was the matter .
[S2] as soon as shiro saw his master he ran to meet him , wagging his tail .
[S3] seizing the end of his kimono , he dragged him under a large yenoki tree .
[S4] here he began to dig very industriously with his paws , yelping with joy all the time .
[S5] the old man , unable to understand what it all meant , stood looking on in bewilderment .
[S6] but shiro went on barking and digging with all his might .

### New Split

[S0] one day shiro was heard barking for a long time in the field at the back of his master 's house .
[S1] the old man , thinking that perhaps some birds were attacking the corn , hurried out to see what was the matter .
[S2] as soon as shiro saw his master he ran to meet him , wagging his tail .
[S3] seizing the end of his kimono , he dragged him under a large yenoki tree .
[S4] here he began to dig very industriously with his paws , yelping with joy all the time .
[S5] the old man , unable to understand what it all meant , stood looking on in bewilderment .
[S6] but shiro went on barking and digging with all his might .

## Sample 37

**story_name:** a-fish-story  
**question:** what happened after the fish tribe gave up on lighting the fire ?  
**answer:** a fish offered to help them .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] ' let me try , ' cried biernuga , the bony fish , but he had no better luck , and no more had kumbal , the bream , nor any of the rest .
[S1] ' it is no use , ' exclaimed thuggai , at last .
[S2] ' the wood is too wet .
[S3] we must just sit and wait till the sun comes out again and dries it .
[S4] ' then a very little fish indeed , not more than four inches long and the youngest of the tribe , bowed himself before thuggai , saying , ' ask my father , guddhu the cod , to light the fire .
[S5] he is skilled in magic more than most fishes .
[S6] ' so thuggai asked him , and guddhu stripped some pieces of bark off a tree , and placed them on top of the smouldering ashes .
[S7] then he knelt by the side of the fire and blew at it for a long while , till slowly the feeble red glow became a little stronger and the edges of the bark showed signs of curling up .
[S8] when the rest of the tribe saw this they pressed close , keeping their backs towards the piercing wind , but guddhu told them they must go to the other side , as he wanted the wind to fan his fire .
[S9] by and by the spark grew into a flame , and a merry crackling was heard .

### New Split

[S0] ' let me try , ' cried biernuga , the bony fish , but he had no better luck , and no more had kumbal , the bream , nor any of the rest .
[S1] ' it is no use , ' exclaimed thuggai , at last .
[S2] ' the wood is too wet . '
[S3] ' we must just sit and wait till the sun comes out again and dries it . '
[S4] then a very little fish indeed , not more than four inches long and the youngest of the tribe , bowed himself before thuggai , saying , ' ask my father , guddhu the cod , to light the fire . '
[S5] ' he is skilled in magic more than most fishes . '
[S6] so thuggai asked him , and guddhu stripped some pieces of bark off a tree , and placed them on top of the smouldering ashes .
[S7] then he knelt by the side of the fire and blew at it for a long while , till slowly the feeble red glow became a little stronger and the edges of the bark showed signs of curling up .
[S8] when the rest of the tribe saw this they pressed close , keeping their backs towards the piercing wind , but guddhu told them they must go to the other side , as he wanted the wind to fan his fire .
[S9] by and by the spark grew into a flame , and a merry crackling was heard .

## Sample 38

**story_name:** naughty-boy  
**question:** why would people outside get wet ?  
**answer:** because there was a rain storm .  

**old sentence count:** 4  
**new sentence count:** 4  

### Old Split

[S0] a long time ago , there lived an old poet , a thoroughly kind old poet .
[S1] as he was sitting one evening in his room , a dreadful storm arose without , and the rain streamed down from heaven .
[S2] but the old poet sat warm and comfortable in his chimney - corner , where the fire blazed and the roasting apple hissed .
[S3] " those who have not a roof over their heads will be wetted to the skin , " said the good old poet .

### New Split

[S0] a long time ago , there lived an old poet , a thoroughly kind old poet .
[S1] as he was sitting one evening in his room , a dreadful storm arose without , and the rain streamed down from heaven .
[S2] but the old poet sat warm and comfortable in his chimney - corner , where the fire blazed and the roasting apple hissed .
[S3] " those who have not a roof over their heads will be wetted to the skin , " said the good old poet .

## Sample 39

**story_name:** the-teapot  
**question:** what was the teapot used for after it broke ?  
**answer:** as a flower pot .  

**old sentence count:** 17  
**new sentence count:** 17  

### Old Split

[S0] all this said the teapot in its fresh young life .
[S1] it stood on the table that was spread for tea ; it was lifted by a very delicate hand , but the delicate hand was awkward .
[S2] the teapot fell , the spout snapped off , and the handle snapped off .
[S3] the lid was no worse to speak of ; the worst had been spoken of that .
[S4] the teapot lay in a swoon on the floor , while the boiling water ran out of it .
[S5] it was a horrid shame , but the worst was that everybody jeered at it ; they jeered at the teapot and not at the awkward hand .
[S6] " i never shall forget that experience , " said the teapot , when it afterward talked of its life .
[S7] " i was called an invalid , and placed in a corner , and the next day was given to a woman who begged for victuals .
[S8] i fell into poverty , and stood dumb both outside and in .
[S9] but then , just as i was , began my better life .
[S10] one can be one thing and still become quite another .
[S11] " earth was placed in me .
[S12] for a teapot , this is the same as being buried , but in the earth was placed a flower bulb .
[S13] who placed it there , who gave it , i know not ; but given it was , and it became a compensation for the chinese leaves and the boiling water , a compensation for the broken handle and spout .
[S14] " and the bulb lay in the earth , the bulb lay in me ; it became my heart , my living heart , such as i had never before possessed .
[S15] there was life in me , power and might .
[S16] the heart pulsed , and the bulb put forth sprouts ; it was the springing up of thoughts and feelings which burst forth into flower .

### New Split

[S0] all this said the teapot in its fresh young life .
[S1] it stood on the table that was spread for tea ; it was lifted by a very delicate hand , but the delicate hand was awkward .
[S2] the teapot fell , the spout snapped off , and the handle snapped off .
[S3] the lid was no worse to speak of ; the worst had been spoken of that .
[S4] the teapot lay in a swoon on the floor , while the boiling water ran out of it .
[S5] it was a horrid shame , but the worst was that everybody jeered at it ; they jeered at the teapot and not at the awkward hand .
[S6] " i never shall forget that experience , " said the teapot , when it afterward talked of its life .
[S7] " i was called an invalid , and placed in a corner , and the next day was given to a woman who begged for victuals . "
[S8] " i fell into poverty , and stood dumb both outside and in . "
[S9] " but then , just as i was , began my better life . "
[S10] " one can be one thing and still become quite another . "
[S11] earth was placed in me .
[S12] for a teapot , this is the same as being buried , but in the earth was placed a flower bulb .
[S13] who placed it there , who gave it , i know not ; but given it was , and it became a compensation for the chinese leaves and the boiling water , a compensation for the broken handle and spout .
[S14] " and the bulb lay in the earth , the bulb lay in me ; it became my heart , my living heart , such as i had never before possessed .
[S15] there was life in me , power and might .
[S16] the heart pulsed , and the bulb put forth sprouts ; it was the springing up of thoughts and feelings which burst forth into flower .

## Sample 40

**story_name:** kings-hares  
**question:** why did the man think peter might not be suited for the shepherd job ?  
**answer:** peter was a sleepy - head .  

**old sentence count:** 11  
**new sentence count:** 11  

### Old Split

[S0] once upon a time there was a man who lived in the little back room .
[S1] he had given up his estate to the heir .
[S2] but in addition he had three sons , who were named peter , paul and esben , who was the youngest .
[S3] all three hung around at home and would not work , for they had it too easy .
[S4] they thought themselves too good for anything like work , and nothing was good enough for them .
[S5] finally peter once heard that the king wanted a shepherd for his hares , and he told his father he would apply for the position , as it would just suit him , seeing that he wished to serve no one lower in rank than the king .
[S6] his father , it is true , was of the opinion that there might be other work that would suit him better , for whoever was to herd hares would have to be quick and spry , and not a sleepy - head .
[S7] when the hares took to their heels in all directions , it was a dance of another kind than when one skipped about a room .
[S8] but it was of no use .
[S9] peter insisted , and would have his own way , took his knapsack , and shambled down hill .
[S10] after he had gone a while , he saw an old woman who had got her nose wedged in a tree - stump while chopping wood , and when peter saw her jerking and pulling away , trying to get out , he burst into loud laughter .

### New Split

[S0] once upon a time there was a man who lived in the little back room .
[S1] he had given up his estate to the heir .
[S2] but in addition he had three sons , who were named peter , paul and esben , who was the youngest .
[S3] all three hung around at home and would not work , for they had it too easy .
[S4] they thought themselves too good for anything like work , and nothing was good enough for them .
[S5] finally peter once heard that the king wanted a shepherd for his hares , and he told his father he would apply for the position , as it would just suit him , seeing that he wished to serve no one lower in rank than the king .
[S6] his father , it is true , was of the opinion that there might be other work that would suit him better , for whoever was to herd hares would have to be quick and spry , and not a sleepy - head .
[S7] when the hares took to their heels in all directions , it was a dance of another kind than when one skipped about a room .
[S8] but it was of no use .
[S9] peter insisted , and would have his own way , took his knapsack , and shambled down hill .
[S10] after he had gone a while , he saw an old woman who had got her nose wedged in a tree - stump while chopping wood , and when peter saw her jerking and pulling away , trying to get out , he burst into loud laughter .

## Sample 41

**story_name:** the-elfin-knight  
**question:** why did the goblins convince earl gregory to get into their ring and throw their spell over him ?  
**answer:** they wanted him to spend many years with them .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] and this cup was filled with heather ale , which foamed up over the brim ; and when the knight saw sir gregory , he lifted it from the table , and handed it to him with a stately bow , and sir gregory , being very thirsty , drank .
[S1] and as he drank he noticed that the ale in the goblet never grew less , but ever foamed up to the edge ; and for the first time his heart misgave him , and he wished that he had never set out on this strange adventure .
[S2] but , alas !
[S3] the time for regrets had passed , for already a strange numbness was stealing over his limbs , and a chill pallor was creeping over his face , and before he could utter a single cry for help the goblet dropped from his nerveless fingers , and he fell down before the elfin king like a dead man .
[S4] then a great shout of triumph went up from all the company ; for if there was one thing which filled their hearts with joy , it was to entice some unwary mortal into their ring and throw their uncanny spell over him , so that he must needs spend long years in their company .

### New Split

[S0] and this cup was filled with heather ale , which foamed up over the brim ; and when the knight saw sir gregory , he lifted it from the table , and handed it to him with a stately bow , and sir gregory , being very thirsty , drank .
[S1] and as he drank he noticed that the ale in the goblet never grew less , but ever foamed up to the edge ; and for the first time his heart misgave him , and he wished that he had never set out on this strange adventure .
[S2] but , alas !
[S3] the time for regrets had passed , for already a strange numbness was stealing over his limbs , and a chill pallor was creeping over his face , and before he could utter a single cry for help the goblet dropped from his nerveless fingers , and he fell down before the elfin king like a dead man .
[S4] then a great shout of triumph went up from all the company ; for if there was one thing which filled their hearts with joy , it was to entice some unwary mortal into their ring and throw their uncanny spell over him , so that he must needs spend long years in their company .

## Sample 42

**story_name:** per-gynt  
**question:** how did per gynt feel when he saw the girl disappear and a pack of bears ?  
**answer:** surprised .  

**old sentence count:** 7  
**new sentence count:** 6  

### Old Split

[S0] in the morning per gynt went out hunting .
[S1] when he had made his way far into the fjoll , he saw a girl driving sheep and goats across a mountain - top .
[S2] but when he reached the top of the mountain , the girl had vanished , as well as her flock , and all he saw was a great pack of bears .
[S3] " never yet have i seen bears run together in packs , " thought per gynt .
[S4] but when he came nearer , they all disappeared save one alone .
[S5] then a voice called from a nearby hill : " guard your boar , for understand , per gynt is without , with his firelock in his hand !
[S6] "

### New Split

[S0] in the morning per gynt went out hunting .
[S1] when he had made his way far into the fjoll , he saw a girl driving sheep and goats across a mountain - top .
[S2] but when he reached the top of the mountain , the girl had vanished , as well as her flock , and all he saw was a great pack of bears .
[S3] " never yet have i seen bears run together in packs , " thought per gynt .
[S4] but when he came nearer , they all disappeared save one alone .
[S5] then a voice called from a nearby hill : " guard your boar , for understand , per gynt is without , with his firelock in his hand ! "

## Sample 43

**story_name:** the-toad  
**question:** what does the little toad think to herself while watching the stars and the moon ?  
**answer:** she decides to go higher up , into splendor and joy .  

**old sentence count:** 22  
**new sentence count:** 21  

### Old Split

[S0] and she was invited to the concert in the evening -- the family concert .
[S1] great enthusiasm and thin voices ; we know the sort of thing .
[S2] no refreshments were given , only there was plenty to drink , for the whole pond was free .
[S3] " now i shall resume my journey , " said the little toad .
[S4] she always felt a longing for something better .
[S5] she saw the stars shining , so large and so bright , and she saw the moon gleaming .
[S6] then she saw the sun rise , and mount higher and higher .
[S7] " perhaps after all , i am still in a well , only in a larger well .
[S8] i must get higher yet ; i feel a great restlessness and longing .
[S9] " and when the moon became round and full , the poor creature thought , " i wonder if that is the bucket which will be let down , and into which i must step to get higher up ?
[S10] or is the sun the great bucket ?
[S11] how great it is !
[S12] how bright it is !
[S13] it can take up all .
[S14] i must look out , that i may not miss the opportunity .
[S15] oh , how it seems to shine in my head !
[S16] i do n't think the jewel can shine brighter .
[S17] but i have n't the jewel ; not that i cry about that -- no , i must go higher up , into splendor and joy !
[S18] i feel so confident , and yet i am afraid .
[S19] it 's a difficult step to take , and yet it must be taken .
[S20] onward , therefore , straight onward !
[S21] "

### New Split

[S0] and she was invited to the concert in the evening -- the family concert .
[S1] great enthusiasm and thin voices ; we know the sort of thing .
[S2] no refreshments were given , only there was plenty to drink , for the whole pond was free .
[S3] " now i shall resume my journey , " said the little toad .
[S4] she always felt a longing for something better .
[S5] she saw the stars shining , so large and so bright , and she saw the moon gleaming .
[S6] then she saw the sun rise , and mount higher and higher .
[S7] " perhaps after all , i am still in a well , only in a larger well . "
[S8] " i must get higher yet ; i feel a great restlessness and longing . "
[S9] and when the moon became round and full , the poor creature thought , " i wonder if that is the bucket which will be let down , and into which i must step to get higher up ? "
[S10] " or is the sun the great bucket ? "
[S11] " how great it is ! "
[S12] " how bright it is ! "
[S13] " it can take up all . "
[S14] " i must look out , that i may not miss the opportunity . "
[S15] " oh , how it seems to shine in my head ! "
[S16] " i do n't think the jewel can shine brighter . "
[S17] " but i have n't the jewel ; not that i cry about that -- no , i must go higher up , into splendor and joy ! "
[S18] " i feel so confident , and yet i am afraid . "
[S19] " it 's a difficult step to take , and yet it must be taken . "
[S20] " onward , therefore , straight onward ! "

## Sample 44

**story_name:** story-of-old-man-who-made-withered-trees-to-flower  
**question:** what will the cross old neighbor do after seeing the coins that the old man found ?  
**answer:** try to find his own treasure .  

**old sentence count:** 8  
**new sentence count:** 7  

### Old Split

[S0] the thought that something might be hidden beneath the tree , and that the dog had scented it , at last struck the old man .
[S1] he ran back to the house , fetched his spade and began to dig the ground at that spot .
[S2] what was his astonishment when , after digging for some time , he came upon a heap of old and valuable coins .
[S3] the deeper he dug the more gold coins did he find .
[S4] so intent was the old man on his work that he never saw the cross face of his neighbor peering at him through the bamboo hedge .
[S5] at last all the gold coins lay shining on the ground .
[S6] shiro sat by erect with pride and looking fondly at his master as if to say , " you see , though only a dog , i can make some return for all the kindness you show me .
[S7] "

### New Split

[S0] the thought that something might be hidden beneath the tree , and that the dog had scented it , at last struck the old man .
[S1] he ran back to the house , fetched his spade and began to dig the ground at that spot .
[S2] what was his astonishment when , after digging for some time , he came upon a heap of old and valuable coins .
[S3] the deeper he dug the more gold coins did he find .
[S4] so intent was the old man on his work that he never saw the cross face of his neighbor peering at him through the bamboo hedge .
[S5] at last all the gold coins lay shining on the ground .
[S6] shiro sat by erect with pride and looking fondly at his master as if to say , " you see , though only a dog , i can make some return for all the kindness you show me . "

## Sample 45

**story_name:** the-red-swan  
**question:** why did the chief's daughter ignore maidwa and the red swan ?  
**answer:** she was jealous of them as a couple .  

**old sentence count:** 18  
**new sentence count:** 18  

### Old Split

[S0] as they went on and came to the lodge of the first old man , their reception and farewell were the same ; and when maidwa glanced to the corner , the silent kettle , which had been the first acquaintance he had made in that family on his travels , was not there .
[S1] the old man smiled when he discovered the direction of maidwa 's glance , but he said nothing .
[S2] when , on continuing their journey , they at last approached the first town which maidwa had passed in his pursuit , the watchman gave notice as before , and he was shown into the chief 's lodge .
[S3] " sit down there , son - in - law , " said the chief , pointing to a place near his daughter .
[S4] " and you also , " he said to the red swan .
[S5] the chief 's daughter was engaged in coloring a girdle , and , as if indifferent to these visitors , she did not even raise her head .
[S6] presently the chief said , " let some one bring in the bundle of our son - in - law .
[S7] " when the bundle was laid before him , maidwa opened one of the bags which had been given to him .
[S8] it was filled with various costly articles -- wampum , robes , and trinkets , of much richness and value ; these , in token of his kindness , he presented to the chief .
[S9] the chief 's daughter stole a glance at the costly gifts , then at maidwa and his beautiful wife .
[S10] she stopped working , and was silent and thoughtful all the evening .
[S11] the chief himself talked with maidwa of his adventures , congratulated him on his good fortune , and concluded by telling him that he should take his daughter along with him in the morning .
[S12] maidwa said " yes .
[S13] " the chief then spoke up , saying , " daughter , be ready to go with him in the morning .
[S14] " now it happened when the chief was thus speaking that there was a foolish fellow in the lodge , who had thought to have got this chief 's daughter for a wife ; and he jumped up , saying : " who is he , " looking grimly at maidwa , " that he should take her for a few presents ?
[S15] i will kill him .
[S16] " and he raised a knife which he had in his hand , and gave it a mighty flourish in the air .
[S17] he kept up this terrible flourish till some one came and pulled him back to his seat , which he had been waiting for , and then he sat quiet enough .

### New Split

[S0] as they went on and came to the lodge of the first old man , their reception and farewell were the same ; and when maidwa glanced to the corner , the silent kettle , which had been the first acquaintance he had made in that family on his travels , was not there .
[S1] the old man smiled when he discovered the direction of maidwa 's glance , but he said nothing .
[S2] when , on continuing their journey , they at last approached the first town which maidwa had passed in his pursuit , the watchman gave notice as before , and he was shown into the chief 's lodge .
[S3] " sit down there , son - in - law , " said the chief , pointing to a place near his daughter .
[S4] " and you also , " he said to the red swan .
[S5] the chief 's daughter was engaged in coloring a girdle , and , as if indifferent to these visitors , she did not even raise her head .
[S6] presently the chief said , " let some one bring in the bundle of our son - in - law . "
[S7] when the bundle was laid before him , maidwa opened one of the bags which had been given to him .
[S8] it was filled with various costly articles -- wampum , robes , and trinkets , of much richness and value ; these , in token of his kindness , he presented to the chief .
[S9] the chief 's daughter stole a glance at the costly gifts , then at maidwa and his beautiful wife .
[S10] she stopped working , and was silent and thoughtful all the evening .
[S11] the chief himself talked with maidwa of his adventures , congratulated him on his good fortune , and concluded by telling him that he should take his daughter along with him in the morning .
[S12] maidwa said " yes . "
[S13] the chief then spoke up , saying , " daughter , be ready to go with him in the morning . "
[S14] now it happened when the chief was thus speaking that there was a foolish fellow in the lodge , who had thought to have got this chief 's daughter for a wife ; and he jumped up , saying : " who is he , " looking grimly at maidwa , " that he should take her for a few presents ? "
[S15] " i will kill him . "
[S16] and he raised a knife which he had in his hand , and gave it a mighty flourish in the air .
[S17] he kept up this terrible flourish till some one came and pulled him back to his seat , which he had been waiting for , and then he sat quiet enough .

## Sample 46

**story_name:** the-one-handed-girl  
**question:** how did the king and queen finally recognize his son ?  
**answer:** heard his voice .  

**old sentence count:** 15  
**new sentence count:** 11  

### Old Split

[S0] ' have you forgotten me so soon ?
[S1] ' he asked .
[S2] at the sound of his voice they gave a cry and ran towards him , and poured out questions as to what had happened , and why he looked like that .
[S3] but the prince did not answer any of them .
[S4] ' how is my wife ?
[S5] ' he said .
[S6] there was a pause .
[S7] then the queen replied : ' she is dead .
[S8] ' ' dead !
[S9] ' he repeated , stepping a little backwards .
[S10] ' and my child ?
[S11] ' ' he is dead too .
[S12] ' the young man stood silent .
[S13] then he said , ' show me their graves .
[S14] '

### New Split

[S0] ' have you forgotten me so soon ? ' he asked .
[S1] at the sound of his voice they gave a cry and ran towards him , and poured out questions as to what had happened , and why he looked like that .
[S2] but the prince did not answer any of them .
[S3] ' how is my wife ? ' he said .
[S4] there was a pause .
[S5] then the queen replied : ' she is dead . '
[S6] ' dead ! ' he repeated , stepping a little backwards .
[S7] ' and my child ? '
[S8] ' he is dead too . '
[S9] the young man stood silent .
[S10] then he said , ' show me their graves . '

## Sample 47

**story_name:** the-wolf-and-the-seven-little-goats  
**question:** why did the little goats not open the door ?  
**answer:** their mother has a delicate and sweet voice and the voice of the wolf is hoarse .  

**old sentence count:** 4  
**new sentence count:** 3  

### Old Split

[S0] it was not long before some one came knocking at the house - door , and crying out , " open the door , my dear children , your mother is come back , and has brought each of you something .
[S1] " but the little kids knew it was the wolf by the hoarse voice .
[S2] " we will not open the door , " cried they ; " you are not our mother , she has a delicate and sweet voice , and your voice is hoarse ; you must be the wolf .
[S3] "

### New Split

[S0] it was not long before some one came knocking at the house - door , and crying out , " open the door , my dear children , your mother is come back , and has brought each of you something . "
[S1] but the little kids knew it was the wolf by the hoarse voice .
[S2] " we will not open the door , " cried they ; " you are not our mother , she has a delicate and sweet voice , and your voice is hoarse ; you must be the wolf . "

## Sample 48

**story_name:** prince-featherhead-and-the-princess-celandine  
**question:** how did the fairy of the beech-woods feel when she saw the king and queen's condition ?  
**answer:** pitiful .  

**old sentence count:** 6  
**new sentence count:** 6  

### Old Split

[S0] now it happened that the fairy of the beech - woods lived in the lovely valley to which chance had led the poor fugitives , and it was she who had , in pity for their forlorn condition , sent the king such good luck to his fishing , and generally taken them under her protection .
[S1] this she was all the more inclined to do as she loved children , and little prince featherhead , who never cried and grew prettier day by day , quite won her heart .
[S2] she made the acquaintance of the king and the queen without at first letting them know that she was a fairy , and they soon took a great fancy to her , and even trusted her with the precious prince , whom she carried off to her palace , where she regaled him with cakes and tarts and every other good thing .
[S3] this was the way she chose of making him fond of her ; but afterwards , as he grew older , she spared no pains in educating and training him as a prince should be trained .
[S4] but unfortunately , in spite of all her care , he grew so vain and frivolous that he quitted his peaceful country life in disgust , and rushed eagerly after all the foolish gaieties of the neighbouring town , where his handsome face and charming manners speedily made him popular .
[S5] the king and queen deeply regretted this alteration in their son , but did not know how to mend matters , since the good old fairy had made him so self - willed .

### New Split

[S0] now it happened that the fairy of the beech - woods lived in the lovely valley to which chance had led the poor fugitives , and it was she who had , in pity for their forlorn condition , sent the king such good luck to his fishing , and generally taken them under her protection .
[S1] this she was all the more inclined to do as she loved children , and little prince featherhead , who never cried and grew prettier day by day , quite won her heart .
[S2] she made the acquaintance of the king and the queen without at first letting them know that she was a fairy , and they soon took a great fancy to her , and even trusted her with the precious prince , whom she carried off to her palace , where she regaled him with cakes and tarts and every other good thing .
[S3] this was the way she chose of making him fond of her ; but afterwards , as he grew older , she spared no pains in educating and training him as a prince should be trained .
[S4] but unfortunately , in spite of all her care , he grew so vain and frivolous that he quitted his peaceful country life in disgust , and rushed eagerly after all the foolish gaieties of the neighbouring town , where his handsome face and charming manners speedily made him popular .
[S5] the king and queen deeply regretted this alteration in their son , but did not know how to mend matters , since the good old fairy had made him so self - willed .

## Sample 49

**story_name:** lawn-dyarrig  
**question:** why did the scarf tighten around the queen ?  
**answer:** the queen was not truthful .  

**old sentence count:** 16  
**new sentence count:** 15  

### Old Split

[S0] " this is the man who conquered the green knight and saved me from terrible valley , " said she to the king of erin ; " this is lawn dyarrig , your son .
[S1] " lawn dyarrig took out the three teeth and put them in his father 's mouth .
[S2] they fitted there perfectly , and grew into their old place .
[S3] the king was satisfied , and as the lady would marry no man but lawn dyarrig , he was the bridegroom .
[S4] " i must give you a present , " said the bride to the queen .
[S5] " here is a beautiful scarf which you are to wear as a girdle this evening .
[S6] " the queen put the scarf round her waist .
[S7] " tell me now , " said the bride to the queen , " who was ur 's father .
[S8] " " what father could he have but his own father , the king of erin ?
[S9] " " tighten , scarf , " said the bride .
[S10] that moment the queen thought that her head was in the sky and the lower half of her body down deep in the earth .
[S11] " oh , my grief and my woe !
[S12] " cried the queen .
[S13] " answer my question in truth , and the scarf will stop squeezing you .
[S14] who was ur 's father ?
[S15] " " the gardener , " said the queen .

### New Split

[S0] " this is the man who conquered the green knight and saved me from terrible valley , " said she to the king of erin ; " this is lawn dyarrig , your son . "
[S1] lawn dyarrig took out the three teeth and put them in his father 's mouth .
[S2] they fitted there perfectly , and grew into their old place .
[S3] the king was satisfied , and as the lady would marry no man but lawn dyarrig , he was the bridegroom .
[S4] " i must give you a present , " said the bride to the queen .
[S5] " here is a beautiful scarf which you are to wear as a girdle this evening . "
[S6] the queen put the scarf round her waist .
[S7] " tell me now , " said the bride to the queen , " who was ur 's father . "
[S8] " what father could he have but his own father , the king of erin ? "
[S9] " tighten , scarf , " said the bride .
[S10] that moment the queen thought that her head was in the sky and the lower half of her body down deep in the earth .
[S11] " oh , my grief and my woe ! " cried the queen .
[S12] " answer my question in truth , and the scarf will stop squeezing you . "
[S13] " who was ur 's father ? "
[S14] " the gardener , " said the queen .

## Sample 50

**story_name:** the-toad-woman  
**question:** why did spirit-iron tell his master about the snakeberry ?  
**answer:** spirit - iron knew the toad - woman could not resist the berries .  

**old sentence count:** 5  
**new sentence count:** 4  

### Old Split

[S0] she was still more angry when they made no answer to her complaint ; but when she found that they were stark dead , and placed in this way to mock her , her fury was very great indeed .
[S1] she ran after the tracks of the young man and his mother as fast as she could ; so fast , indeed , that she was on the very point of overtaking them , when the dog , spirit - iron , coming close up to his master , whispered to him--"snakeberry !
[S2] " " let the snakeberry spring up to detain her !
[S3] " cried out the young man ; and immediately the berries spread like scarlet all over the path , for a long distance ; and the old toad - woman , who was almost as fond of these berries as she was of fat bears , could not avoid stooping down to pick and eat .
[S4] the old toad - woman was very anxious to get forward , but the snakeberry - vines kept spreading out on every side ; and they still grow and grow , and spread and spread ; and to this day the wicked old toad - woman is busy picking the berries , and she will never be able to get beyond to the other side , to disturb the happiness of the young hunter and his mother , who still live , with their faithful dog , in the shadow of the beautiful wood - side where they were born .

### New Split

[S0] she was still more angry when they made no answer to her complaint ; but when she found that they were stark dead , and placed in this way to mock her , her fury was very great indeed .
[S1] she ran after the tracks of the young man and his mother as fast as she could ; so fast , indeed , that she was on the very point of overtaking them , when the dog , spirit - iron , coming close up to his master , whispered to him-- " snakeberry ! "
[S2] " let the snakeberry spring up to detain her ! " cried out the young man ; and immediately the berries spread like scarlet all over the path , for a long distance ; and the old toad - woman , who was almost as fond of these berries as she was of fat bears , could not avoid stooping down to pick and eat .
[S3] the old toad - woman was very anxious to get forward , but the snakeberry - vines kept spreading out on every side ; and they still grow and grow , and spread and spread ; and to this day the wicked old toad - woman is busy picking the berries , and she will never be able to get beyond to the other side , to disturb the happiness of the young hunter and his mother , who still live , with their faithful dog , in the shadow of the beautiful wood - side where they were born .

## Sample 51

**story_name:** black-sheep  
**question:** what happened because mamma is so poor ?  
**answer:** she can not buy the boy a new coat .  

**old sentence count:** 7  
**new sentence count:** 6  

### Old Split

[S0] " what will they do with it , black sheep ?
[S1] " enquired the little boy .
[S2] " they will make coats of it , to keep themselves warm .
[S3] " " i wish i had some wool , " said the boy , " for i need a new coat very badly , and mamma is so poor she can not buy me one .
[S4] " " that is too bad , " replied the black sheep ; " but i shall have more wool by and by , and then i will give you a bagful to make a new coat from .
[S5] " " will you really ?
[S6] " asked the boy , looking very much pleased .

### New Split

[S0] " what will they do with it , black sheep ? "
[S1] enquired the little boy .
[S2] " they will make coats of it , to keep themselves warm . "
[S3] " i wish i had some wool , " said the boy , " for i need a new coat very badly , and mamma is so poor she can not buy me one . "
[S4] " that is too bad , " replied the black sheep ; " but i shall have more wool by and by , and then i will give you a bagful to make a new coat from . "
[S5] " will you really ? " asked the boy , looking very much pleased .

## Sample 52

**story_name:** silverwhite-lillwacker  
**question:** what happened to the dogs because silverwhite laid the three troll hairs on his dogs' heads ?  
**answer:** the dogs fell silent and lay motionless as though they had grown fast to the ground .  

**old sentence count:** 9  
**new sentence count:** 9  

### Old Split

[S0] but when the troll noticed he was getting the worst of it , he grew frightened , quickly ran to a high tree , and clambered into it .
[S1] silverwhite and the dogs ran after him , the dogs barking as loudly as they could .
[S2] then the troll begged for his life and said : " dear silverwhite , i will take wergild for my brothers , only bid your dogs be still , so that we may talk .
[S3] " the king bade his dogs be still , but in vain , they only barked the more loudly .
[S4] then the troll tore three hairs from his head , handed them to silverwhite and said : " lay a hair on each of the dogs , and then they will be as quiet as can be .
[S5] " the king did so and at once the dogs fell silent , and lay motionless as though they had grown fast to the ground .
[S6] now silverwhite realized that he had been deceived ; but it was too late .
[S7] the troll was already descending from the tree , and he drew his sword and again began to fight .
[S8] but they had exchanged no more than a few blows , before silverwhite received a mortal wound , and lay on the earth in a pool of blood .

### New Split

[S0] but when the troll noticed he was getting the worst of it , he grew frightened , quickly ran to a high tree , and clambered into it .
[S1] silverwhite and the dogs ran after him , the dogs barking as loudly as they could .
[S2] then the troll begged for his life and said : " dear silverwhite , i will take wergild for my brothers , only bid your dogs be still , so that we may talk . "
[S3] the king bade his dogs be still , but in vain , they only barked the more loudly .
[S4] then the troll tore three hairs from his head , handed them to silverwhite and said : " lay a hair on each of the dogs , and then they will be as quiet as can be . "
[S5] the king did so and at once the dogs fell silent , and lay motionless as though they had grown fast to the ground .
[S6] now silverwhite realized that he had been deceived ; but it was too late .
[S7] the troll was already descending from the tree , and he drew his sword and again began to fight .
[S8] but they had exchanged no more than a few blows , before silverwhite received a mortal wound , and lay on the earth in a pool of blood .

## Sample 53

**story_name:** the-magic-bon-bons  
**question:** how did claribel feel when she realized the senator took her bonbons ?  
**answer:** angry .  

**old sentence count:** 6  
**new sentence count:** 5  

### Old Split

[S0] suddenly claribel sudds , who happened to be present , uttered a scream and sprang to her feet .
[S1] pointing an accusing finger at the dancing senator , she cried in a loud voice : " that 's the man who stole my bonbons !
[S2] seize him !
[S3] arrest him !
[S4] do n't let him escape !
[S5] "

### New Split

[S0] suddenly claribel sudds , who happened to be present , uttered a scream and sprang to her feet .
[S1] pointing an accusing finger at the dancing senator , she cried in a loud voice : " that 's the man who stole my bonbons ! "
[S2] " seize him ! "
[S3] " arrest him ! "
[S4] " do n't let him escape ! "

## Sample 54

**story_name:** storm-magic  
**question:** how did the captain and crew feel about the cabin-boy taking charge ?  
**answer:** ridiculous .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] when the day came on which the cabin - boy was to take command , the weather was fair and quiet ; but he drummed up the whole ship 's crew , and with the exception of a tiny bit of canvas , had all sails reefed .
[S1] the captain and crew laughed at him , and said : " that shows the sort of a captain we have now .
[S2] do n't you want us to reef that last bit of sail this very minute ?
[S3] " " not yet , " answered the cabin - boy , " but before long .
[S4] " suddenly a squall struck them , struck them so heavily that they thought they would capsize , and had they not reefed the sails they would undoubtedly have foundered when the first breaker roared down upon the ship .

### New Split

[S0] when the day came on which the cabin - boy was to take command , the weather was fair and quiet ; but he drummed up the whole ship 's crew , and with the exception of a tiny bit of canvas , had all sails reefed .
[S1] the captain and crew laughed at him , and said : " that shows the sort of a captain we have now . "
[S2] " do n't you want us to reef that last bit of sail this very minute ? "
[S3] " not yet , " answered the cabin - boy , " but before long . "
[S4] suddenly a squall struck them , struck them so heavily that they thought they would capsize , and had they not reefed the sails they would undoubtedly have foundered when the first breaker roared down upon the ship .

## Sample 55

**story_name:** story-of-man-who-did-not-wish-to-die  
**question:** how did jofuku know sentaro's desire for death was not real ?  
**answer:** he cried loudly for help when he thought the shark would eat him .  

**old sentence count:** 6  
**new sentence count:** 5  

### Old Split

[S0] suddenly a bright light came towards him , and in the light stood a messenger .
[S1] the messenger held a book in his hand , and spoke to sentaro : " i am sent to you by jofuku , who in answer to your prayer , has permitted you in a dream to see the land of perpetual life .
[S2] but you grew weary of living there , and begged to be allowed to return to your native land so that you might die .
[S3] jofuku , so that he might try you , allowed you to drop into the sea , and then sent a shark to swallow you up .
[S4] your desire for death was not real , for even at that moment you cried out loudly and shouted for help .
[S5] "

### New Split

[S0] suddenly a bright light came towards him , and in the light stood a messenger .
[S1] the messenger held a book in his hand , and spoke to sentaro : " i am sent to you by jofuku , who in answer to your prayer , has permitted you in a dream to see the land of perpetual life . "
[S2] " but you grew weary of living there , and begged to be allowed to return to your native land so that you might die . "
[S3] " jofuku , so that he might try you , allowed you to drop into the sea , and then sent a shark to swallow you up . "
[S4] " your desire for death was not real , for even at that moment you cried out loudly and shouted for help . "

## Sample 56

**story_name:** the-uraschimataro-and-the-turtle  
**question:** how did uraschimataro feel when he struggled hard to reach the shore ?  
**answer:** scared .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] years flew by , and every morning uraschimataro sailed his boat into the deep sea .
[S1] but one day as he was making for a little bay between some rocks , there arose a fierce whirlwind , which shattered his boat to pieces , and she was sucked under by the waves .
[S2] uraschimataro himself very nearly shared the same fate .
[S3] but he was a powerful swimmer , and struggled hard to reach the shore .
[S4] then he saw a large turtle coming towards him , and above the howling of the storm he heard what it said : ' i am the turtle whose life you once saved .
[S5] i will now pay my debt and show my gratitude .
[S6] the land is still far distant , and without my help you would never get there .
[S7] climb on my back , and i will take you where you will .
[S8] ' uraschimataro did not wait to be asked twice , and thankfully accepted his friend 's help .
[S9] but scarcely was he seated firmly on the shell , when the turtle proposed that they should not return to the shore at once , but go under the sea , and look at some of the wonders that lay hidden there .

### New Split

[S0] years flew by , and every morning uraschimataro sailed his boat into the deep sea .
[S1] but one day as he was making for a little bay between some rocks , there arose a fierce whirlwind , which shattered his boat to pieces , and she was sucked under by the waves .
[S2] uraschimataro himself very nearly shared the same fate .
[S3] but he was a powerful swimmer , and struggled hard to reach the shore .
[S4] then he saw a large turtle coming towards him , and above the howling of the storm he heard what it said : ' i am the turtle whose life you once saved . '
[S5] ' i will now pay my debt and show my gratitude . '
[S6] ' the land is still far distant , and without my help you would never get there . '
[S7] ' climb on my back , and i will take you where you will . '
[S8] uraschimataro did not wait to be asked twice , and thankfully accepted his friend 's help .
[S9] but scarcely was he seated firmly on the shell , when the turtle proposed that they should not return to the shore at once , but go under the sea , and look at some of the wonders that lay hidden there .

## Sample 57

**story_name:** how-brave-walter-hunted-wolves  
**question:** what did walter offer jonas if he went with him ?  
**answer:** the wolf 's skin .  

**old sentence count:** 10  
**new sentence count:** 8  

### Old Split

[S0] ' that 's a lie , ' said walter , ' i am not at all frightened , but it is more amusing when there are two .
[S1] i only want someone who will see how i strike the wolf and how the dust flies out of his skin .
[S2] ' ' well , then , walter can take the miller 's little lisa with him .
[S3] she can sit on a stone and look on , ' said jonas .
[S4] ' no , she would certainly be frightened , ' said walter , ' and how would it do for a girl to go wolf - hunting ?
[S5] come with me , jonas , and you shall have the skin , and i will be content with the ears and the tail .
[S6] ' ' no , thank you , ' said jonas , ' walter can keep the skin for himself .
[S7] now i see quite well that he is frightened .
[S8] fie , shame on him !
[S9] '

### New Split

[S0] ' that 's a lie , ' said walter , ' i am not at all frightened , but it is more amusing when there are two . '
[S1] ' i only want someone who will see how i strike the wolf and how the dust flies out of his skin . '
[S2] ' well , then , walter can take the miller 's little lisa with him . she can sit on a stone and look on , ' said jonas .
[S3] ' no , she would certainly be frightened , ' said walter , ' and how would it do for a girl to go wolf - hunting ? '
[S4] ' come with me , jonas , and you shall have the skin , and i will be content with the ears and the tail . '
[S5] ' no , thank you , ' said jonas , ' walter can keep the skin for himself . '
[S6] ' now i see quite well that he is frightened . '
[S7] ' fie , shame on him ! '

## Sample 58

**story_name:** brave-tin-soldier  
**question:** why did the tin soldier think that the little lady had one leg ?  
**answer:** because she was dancing .  

**old sentence count:** 4  
**new sentence count:** 4  

### Old Split

[S0] the little lady was a dancer , and she stretched out both her arms , and raised one of her legs so high , that the tin soldier could not see it at all , and he thought that she , like himself , had only one leg .
[S1] " that is the wife for me , " he thought ; " but she is too grand , and lives in a castle , while i have only a box to live in , five - and - twenty of us altogether , that is no place for her .
[S2] still i must try and make her acquaintance .
[S3] " then he laid himself at full length on the table behind a snuff - box that stood upon it , so that he could peep at the little delicate lady , who continued to stand on one leg without losing her balance .

### New Split

[S0] the little lady was a dancer , and she stretched out both her arms , and raised one of her legs so high , that the tin soldier could not see it at all , and he thought that she , like himself , had only one leg .
[S1] " that is the wife for me , " he thought ; " but she is too grand , and lives in a castle , while i have only a box to live in , five - and - twenty of us altogether , that is no place for her . "
[S2] " still i must try and make her acquaintance . "
[S3] then he laid himself at full length on the table behind a snuff - box that stood upon it , so that he could peep at the little delicate lady , who continued to stand on one leg without losing her balance .

## Sample 59

**story_name:** the-brownie-of-the-lake  
**question:** why was barbaik furious ?  
**answer:** each morning , she milk the cows and go to market . each evening , she had to sit up till midnight to churn the butter .  

**old sentence count:** 3  
**new sentence count:** 3  

### Old Split

[S0] barbaik was furious .
[S1] each morning when she was obliged to get up before dawn to milk the cows and go to market , and each evening when she had to sit up till midnight in order to churn the butter , her heart was filled with rage against the brownie who had caused her to expect a life of ease and pleasure .
[S2] but when she looked at jegu and beheld his red face , squinting eyes , and untidy hair , her anger was doubled .

### New Split

[S0] barbaik was furious .
[S1] each morning when she was obliged to get up before dawn to milk the cows and go to market , and each evening when she had to sit up till midnight in order to churn the butter , her heart was filled with rage against the brownie who had caused her to expect a life of ease and pleasure .
[S2] but when she looked at jegu and beheld his red face , squinting eyes , and untidy hair , her anger was doubled .

## Sample 60

**story_name:** the-ugly-duckling  
**question:** how did the duckling feel when he realized he was a swan himself ?  
**answer:** happy .  

**old sentence count:** 6  
**new sentence count:** 6  

### Old Split

[S0] " i will fly to them , these royal birds , and they will peck me to death because i , who am so ugly , dare to approach them ; but it does n't matter ; it 's better to be killed by them than to be snapped at by the ducks and pecked at by hens and kicked by the servant who looks after the poultry - yard , and suffer all the winter .
[S1] " so he flew out into the open water and swam towards the stately swans , and they saw him and hastened with swelling plumage to meet him .
[S2] " yes , kill me , " the poor creature said , bowing his head down to the water , and waited for death .
[S3] but what did he see in the clear water ?
[S4] he beheld his own image , but it was no longer that of a clumsy dark grey bird , ugly and repulsive .
[S5] he was a swan himself .

### New Split

[S0] " i will fly to them , these royal birds , and they will peck me to death because i , who am so ugly , dare to approach them ; but it does n't matter ; it 's better to be killed by them than to be snapped at by the ducks and pecked at by hens and kicked by the servant who looks after the poultry - yard , and suffer all the winter . "
[S1] so he flew out into the open water and swam towards the stately swans , and they saw him and hastened with swelling plumage to meet him .
[S2] " yes , kill me , " the poor creature said , bowing his head down to the water , and waited for death .
[S3] but what did he see in the clear water ?
[S4] he beheld his own image , but it was no longer that of a clumsy dark grey bird , ugly and repulsive .
[S5] he was a swan himself .

## Sample 61

**story_name:** brother-sister  
**question:** how did the sister feel when the fawn leaved for the hunt ?  
**answer:** worried .  

**old sentence count:** 10  
**new sentence count:** 9  

### Old Split

[S0] but the sister was very terrified when she saw that her fawn was wounded .
[S1] she washed his foot , laid cooling leaves round it , and said , " lie down on your bed , dear fawn , and rest , that you may be soon well .
[S2] " the wound was very slight , so that the fawn felt nothing of it the next morning .
[S3] and when he heard the noise of the hunting outside , he said , " i can not stay in , i must go after them ; i shall not be taken easily again !
[S4] " the sister began to weep , and said , " i know you will be killed , and i left alone here in the forest , and forsaken of everybody .
[S5] i can not let you go !
[S6] " " then i shall die here with longing , " answered the fawn ; " when i hear the sound of the horn i feel as if i should leap out of my skin .
[S7] " then the sister , seeing there was no help for it , unlocked the door with a heavy heart , and the fawn bounded away into the forest , well and merry .
[S8] when the king saw him , he said to his hunters , " now , follow him up all day long till the night comes , and see that you do him no hurt .
[S9] "

### New Split

[S0] but the sister was very terrified when she saw that her fawn was wounded .
[S1] she washed his foot , laid cooling leaves round it , and said , " lie down on your bed , dear fawn , and rest , that you may be soon well . "
[S2] the wound was very slight , so that the fawn felt nothing of it the next morning .
[S3] and when he heard the noise of the hunting outside , he said , " i can not stay in , i must go after them ; i shall not be taken easily again ! "
[S4] the sister began to weep , and said , " i know you will be killed , and i left alone here in the forest , and forsaken of everybody . "
[S5] " i can not let you go ! "
[S6] " then i shall die here with longing , " answered the fawn ; " when i hear the sound of the horn i feel as if i should leap out of my skin . "
[S7] then the sister , seeing there was no help for it , unlocked the door with a heavy heart , and the fawn bounded away into the forest , well and merry .
[S8] when the king saw him , he said to his hunters , " now , follow him up all day long till the night comes , and see that you do him no hurt . "

## Sample 62

**story_name:** the-flying-ogre  
**question:** what will the monk do when the stranger tells him to say the truth ?  
**answer:** point to the hollow tree .  

**old sentence count:** 19  
**new sentence count:** 18  

### Old Split

[S0] " have you seen the girl in the red coat ?
[S1] " asked the stranger .
[S2] and when the monk replied that he had seen nothing , the other continued : " bonze , you should not lie !
[S3] this girl is not a human being , but a flying ogre .
[S4] of flying ogres there are thousands of varieties , who bring ruin to people everywhere .
[S5] i have already slain a countless number of them , and have pretty well done away with them .
[S6] but this one is the worst of all .
[S7] last night the lord of the heavens gave me a triple command , and that is the reason i have hurried down from the skies .
[S8] there are eight thousand of us under way in all directions to catch this monster .
[S9] if you do not tell the truth , monk , then you are sinning against heaven itself !
[S10] " upon that the monk did not dare deceive him , but pointed to the hollow tree .
[S11] the messenger of the skies dismounted , stepped into the tree and looked about him .
[S12] then he once more mounted his horse , which carried him up the hollow trunk and out at the end of the tree .
[S13] the monk looked up and could see a small , red flame come out of the tree - top .
[S14] it was followed by the messenger of the skies .
[S15] both rose up to the clouds and disappeared .
[S16] after a time there fell a rain of blood .
[S17] the ogre had probably been hit by an arrow or captured .
[S18] afterward the monk told the tale to the scholar who wrote it down .

### New Split

[S0] " have you seen the girl in the red coat ? " asked the stranger .
[S1] and when the monk replied that he had seen nothing , the other continued : " bonze , you should not lie ! "
[S2] " this girl is not a human being , but a flying ogre . "
[S3] " of flying ogres there are thousands of varieties , who bring ruin to people everywhere . "
[S4] " i have already slain a countless number of them , and have pretty well done away with them . "
[S5] " but this one is the worst of all . "
[S6] " last night the lord of the heavens gave me a triple command , and that is the reason i have hurried down from the skies . "
[S7] " there are eight thousand of us under way in all directions to catch this monster . "
[S8] " if you do not tell the truth , monk , then you are sinning against heaven itself ! "
[S9] upon that the monk did not dare deceive him , but pointed to the hollow tree .
[S10] the messenger of the skies dismounted , stepped into the tree and looked about him .
[S11] then he once more mounted his horse , which carried him up the hollow trunk and out at the end of the tree .
[S12] the monk looked up and could see a small , red flame come out of the tree - top .
[S13] it was followed by the messenger of the skies .
[S14] both rose up to the clouds and disappeared .
[S15] after a time there fell a rain of blood .
[S16] the ogre had probably been hit by an arrow or captured .
[S17] afterward the monk told the tale to the scholar who wrote it down .

## Sample 63

**story_name:** storm-magic  
**question:** what will be the worst squall ?  
**answer:** the last one .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] " that 's all very well , but we are not through yet , " said the boy , " there is worse to come , " and he told them to reef every last rag , as well as what had been left of the topsails .
[S1] the second squall hit them with even greater force than the first , and was so vicious and violent that the whole crew was frightened .
[S2] while it was at its worst , the boy told them to throw overboard the second cord ; and they threw it over billet by billet , and took care not to take any from the third cord .
[S3] when the last billet fell , they again heard a deep groan , and then all was still .
[S4] " now there will be one more squall , and that will be the worst , " said the boy , and sent every one to his station .
[S5] there was not a hawser loose on the whole ship .
[S6] the last squall hit them with far more force than either of the preceding ones , the ship laid over on her side so that they thought she would not right herself again , and the breaker swept over the deck .

### New Split

[S0] " that 's all very well , but we are not through yet , " said the boy , " there is worse to come , " and he told them to reef every last rag , as well as what had been left of the topsails .
[S1] the second squall hit them with even greater force than the first , and was so vicious and violent that the whole crew was frightened .
[S2] while it was at its worst , the boy told them to throw overboard the second cord ; and they threw it over billet by billet , and took care not to take any from the third cord .
[S3] when the last billet fell , they again heard a deep groan , and then all was still .
[S4] " now there will be one more squall , and that will be the worst , " said the boy , and sent every one to his station .
[S5] there was not a hawser loose on the whole ship .
[S6] the last squall hit them with far more force than either of the preceding ones , the ship laid over on her side so that they thought she would not right herself again , and the breaker swept over the deck .

## Sample 64

**story_name:** thomas-the-rhymer  
**question:** why did thomas, the snow-white hard, and hind disappear in the river's foaming waters ?  
**answer:** they went to fairy - land .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] when he heard the boy 's message , the seer 's face grew grave and wrapt .
[S1] " it is a summons , " he said softly , " a summons from the queen of fairy - land .
[S2] i have waited long for it , and it hath come at last .
[S3] " and when he went out , instead of joining the little company of waiting men , he walked straight up to the snow - white hart and hind .
[S4] as soon as he reached them they paused for a moment as if to greet him .
[S5] then all three moved slowly down a steep bank that sloped to the little river leader , and disappeared in its foaming waters , for the stream was in full flood .
[S6] and , although a careful search was made , no trace of thomas of ercildoune was found ; and to this day the country folk believe that the hart and the hind were messengers from the elfin queen , and that he went back to fairy - land with them .

### New Split

[S0] when he heard the boy 's message , the seer 's face grew grave and wrapt .
[S1] " it is a summons , " he said softly , " a summons from the queen of fairy - land . "
[S2] " i have waited long for it , and it hath come at last . "
[S3] and when he went out , instead of joining the little company of waiting men , he walked straight up to the snow - white hart and hind .
[S4] as soon as he reached them they paused for a moment as if to greet him .
[S5] then all three moved slowly down a steep bank that sloped to the little river leader , and disappeared in its foaming waters , for the stream was in full flood .
[S6] and , although a careful search was made , no trace of thomas of ercildoune was found ; and to this day the country folk believe that the hart and the hind were messengers from the elfin queen , and that he went back to fairy - land with them .

## Sample 65

**story_name:** rose-of-evening  
**question:** why did aduan's illness leave him ?  
**answer:** he met rose of evening at the lotus field .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] so the boy led him to the south .
[S1] there he opened a gate and they turned the corner , to the west .
[S2] once more the doors of the gate flew open , and now aduan saw a lotus field about twenty acres in size .
[S3] the lotus flowers were all growing on level earth , and their leaves were as large as mats and their flowers like umbrellas .
[S4] the fallen blossoms covered the ground beneath the stalks to the depth of a foot or more .
[S5] the boy led aduan in and said , " now first of all sit down for a little while !
[S6] " then he went away .

### New Split

[S0] so the boy led him to the south .
[S1] there he opened a gate and they turned the corner , to the west .
[S2] once more the doors of the gate flew open , and now aduan saw a lotus field about twenty acres in size .
[S3] the lotus flowers were all growing on level earth , and their leaves were as large as mats and their flowers like umbrellas .
[S4] the fallen blossoms covered the ground beneath the stalks to the depth of a foot or more .
[S5] the boy led aduan in and said , " now first of all sit down for a little while ! "
[S6] then he went away .

## Sample 66

**story_name:** a-legend-of-confucius  
**question:** how did confucius give evidence that he had foreknowledge of many things ?  
**answer:** he had a tablet that predicted emperor tsin schi huang 's actions .  

**old sentence count:** 12  
**new sentence count:** 12  

### Old Split

[S0] when confucius came to the earth , the kilin , that strange beast which is the prince of all four - footed animals , and only appears when there is a great man on earth , sought the child and spat out a jade whereon was written : " son of the watercrystal you are destined to become an uncrowned king !
[S1] " and confucius grew up , studied diligently , learned wisdom and came to be a saint .
[S2] he did much good on earth , and ever since his death has been reverenced as the greatest of teachers and masters .
[S3] he had foreknowledge of many things .
[S4] and even after he had died he gave evidence of this .
[S5] to the left of the coffin was a door , which led into an inner chamber .
[S6] in this chamber stood a bed , and a table with books and clothing , all as though meant for the use of a living person .
[S7] tsin schi huang seated himself on the bed and looked down .
[S8] and there on the floor stood two shoes of red silk , whose tips were adorned with a woven pattern of clouds .
[S9] a bamboo staff leaned against the wall .
[S10] the emperor , in jest , put on the shoes , took the staff and left the grave .
[S11] but as he did so a tablet suddenly appeared before his eyes on which stood the following lines : o'er kingdoms six tsin schi huang his army led , to ope my grave and find my humble bed ; he steals my shoes and takes my staff away to reach schakiu -- and his last earthly day !

### New Split

[S0] when confucius came to the earth , the kilin , that strange beast which is the prince of all four - footed animals , and only appears when there is a great man on earth , sought the child and spat out a jade whereon was written : " son of the watercrystal you are destined to become an uncrowned king ! "
[S1] and confucius grew up , studied diligently , learned wisdom and came to be a saint .
[S2] he did much good on earth , and ever since his death has been reverenced as the greatest of teachers and masters .
[S3] he had foreknowledge of many things .
[S4] and even after he had died he gave evidence of this .
[S5] to the left of the coffin was a door , which led into an inner chamber .
[S6] in this chamber stood a bed , and a table with books and clothing , all as though meant for the use of a living person .
[S7] tsin schi huang seated himself on the bed and looked down .
[S8] and there on the floor stood two shoes of red silk , whose tips were adorned with a woven pattern of clouds .
[S9] a bamboo staff leaned against the wall .
[S10] the emperor , in jest , put on the shoes , took the staff and left the grave .
[S11] but as he did so a tablet suddenly appeared before his eyes on which stood the following lines : o'er kingdoms six tsin schi huang his army led , to ope my grave and find my humble bed ; he steals my shoes and takes my staff away to reach schakiu -- and his last earthly day !

## Sample 67

**story_name:** neighbor-underground  
**question:** how did the neighbor feel about the placement of the peasant's stable ?  
**answer:** unhappy .  

**old sentence count:** 11  
**new sentence count:** 10  

### Old Split

[S0] one day as he was standing in his stable , he sank through the ground .
[S1] down below , in the place to which he had come , everything was unspeakably handsome .
[S2] there was nothing which was not of gold or of silver .
[S3] then the man who had called himself his neighbor came along , and bade him sit down .
[S4] after a time food was brought in on a silver platter , and mead in a silver jug , and the neighbor invited him to draw up to the table and eat .
[S5] the peasant did not dare refuse , and sat down at the table ; but just as he was about to dip his spoon into the dish , something fell down into his food from above , so that he lost his appetite .
[S6] " yes , yes , " said the man , " now you can see why we do n't like your stable .
[S7] we can never eat in peace , for as soon as we sit down to a meal , dirt and straw fall down , and no matter how hungry we may be , we lose our appetites and can not eat .
[S8] but if you will do me the favor to set up your stable elsewhere , you shall never go short of pasture nor good crops , no matter how old you may grow to be .
[S9] but if you wo n't , you shall know naught but lean years all your life long .
[S10] "

### New Split

[S0] one day as he was standing in his stable , he sank through the ground .
[S1] down below , in the place to which he had come , everything was unspeakably handsome .
[S2] there was nothing which was not of gold or of silver .
[S3] then the man who had called himself his neighbor came along , and bade him sit down .
[S4] after a time food was brought in on a silver platter , and mead in a silver jug , and the neighbor invited him to draw up to the table and eat .
[S5] the peasant did not dare refuse , and sat down at the table ; but just as he was about to dip his spoon into the dish , something fell down into his food from above , so that he lost his appetite .
[S6] " yes , yes , " said the man , " now you can see why we do n't like your stable . "
[S7] " we can never eat in peace , for as soon as we sit down to a meal , dirt and straw fall down , and no matter how hungry we may be , we lose our appetites and can not eat . "
[S8] " but if you will do me the favor to set up your stable elsewhere , you shall never go short of pasture nor good crops , no matter how old you may grow to be . "
[S9] " but if you wo n't , you shall know naught but lean years all your life long . "

## Sample 68

**story_name:** the-bracket-bull  
**question:** what will happen when the champion meets the dragon ?  
**answer:** the champion will defeat the dragon .  

**old sentence count:** 13  
**new sentence count:** 12  

### Old Split

[S0] he turned the dragon out into the sea at last .
[S1] he went away then , and said that he would return the next day .
[S2] he left the steed again in the place where he found it , and he took the fine suit off him , and when the other people returned he was before them .
[S3] when the people came home that night they were all talking and saying that some champion came to fight with the dragon and turned him out into the sea again .
[S4] that was the story that every person had , but they did not know who was the champion who did it .
[S5] the next day , when his master and the other people were gone , he went to the castle of the three giants again , and he took out another steed and another suit of valour ( i.e.
[S6] , armour ) , and he brought with him the second giant 's sword , and he went to the place where the dragon was to come .
[S7] the king 's daughter was bound to a post on the shore , waiting for him , and the eyes going out on her head looking would she see the champion coming who fought the dragon the day before .
[S8] there were twice as many people in it as there were on the first day , and they were all waiting till they would see the champion coming .
[S9] when the dragon came the lad went in face of him , and the dragon was half confused and sickened after the fight that he had made the day before .
[S10] they were beating one another till the evening , and then he drove away the dragon .
[S11] the people tried to keep him , but they were not able .
[S12] he went from them .

### New Split

[S0] he turned the dragon out into the sea at last .
[S1] he went away then , and said that he would return the next day .
[S2] he left the steed again in the place where he found it , and he took the fine suit off him , and when the other people returned he was before them .
[S3] when the people came home that night they were all talking and saying that some champion came to fight with the dragon and turned him out into the sea again .
[S4] that was the story that every person had , but they did not know who was the champion who did it .
[S5] the next day , when his master and the other people were gone , he went to the castle of the three giants again , and he took out another steed and another suit of valour ( i.e. , armour ) , and he brought with him the second giant 's sword , and he went to the place where the dragon was to come .
[S6] the king 's daughter was bound to a post on the shore , waiting for him , and the eyes going out on her head looking would she see the champion coming who fought the dragon the day before .
[S7] there were twice as many people in it as there were on the first day , and they were all waiting till they would see the champion coming .
[S8] when the dragon came the lad went in face of him , and the dragon was half confused and sickened after the fight that he had made the day before .
[S9] they were beating one another till the evening , and then he drove away the dragon .
[S10] the people tried to keep him , but they were not able .
[S11] he went from them .

## Sample 69

**story_name:** east-of-sun-and-west-of-moon  
**question:** what happened because the maiden looked at the prince and he awoke ?  
**answer:** he must return to his step - mother 's castle .  

**old sentence count:** 29  
**new sentence count:** 27  

### Old Split

[S0] when they reached home , and the maiden had gone to bed , all went as usual : a man came in and cast himself down in a corner of the room .
[S1] but in the night , when she heard him sleeping soundly , she stood up and lighted the candle .
[S2] she threw the light on him , and saw the handsomest prince one might wish to see .
[S3] and she liked him so exceedingly well that she thought she would be unable to keep on living if she could not kiss him that very minute .
[S4] she did so , but by mistake she let three hot drops of tallow fall on him , and he awoke .
[S5] " alas , what have you done !
[S6] " cried he .
[S7] " now you have made both of us unhappy .
[S8] if you had only held out until the end of the year , i would have been delivered .
[S9] i have a step - mother who has cast a spell on me , so that by day i am a bear , and at night a human being .
[S10] but now all is over between us , and i must return to my step - mother .
[S11] she lives in a castle that is east of the sun and west of the moon , where there is a princess with a nose three yards long , whom i must now marry .
[S12] " the maiden wept and wailed to no avail , for the prince said he must journey away .
[S13] then she asked him whether she might not go with him .
[S14] no , said he , that could not be .
[S15] " but can you not at least tell me the road , so that i can search for you .
[S16] for surely that will be permitted me ?
[S17] " " yes , that you may do , " said he .
[S18] " but there is no road that leads there .
[S19] the castle lies east of the sun and west of the moon , and neither now nor at any other time will you find the road to it !
[S20] " when the maiden awoke the next morning , the prince as well as the castle had disappeared .
[S21] she lay in a green opening in the midst of a thick , dark wood , and beside her lay the bundle of poor belongings she had brought from home .
[S22] and when she had rubbed the sleep out of her eyes , and had cried her fill , she set out and wandered many , many days , until at last she came to a great hill .
[S23] and before the hill sat an old woman who was playing with a golden apple .
[S24] the maiden asked the woman whether she did not know which road led to the prince who lived in the castle that was east of the sun and west of the moon , and who was to marry a princess with a nose three yards long .
[S25] " how do you come to know him ?
[S26] " asked the woman .
[S27] " are you , perhaps , the maiden he wanted to marry ?
[S28] " " yes , i am that maiden , " she replied .

### New Split

[S0] when they reached home , and the maiden had gone to bed , all went as usual : a man came in and cast himself down in a corner of the room .
[S1] but in the night , when she heard him sleeping soundly , she stood up and lighted the candle .
[S2] she threw the light on him , and saw the handsomest prince one might wish to see .
[S3] and she liked him so exceedingly well that she thought she would be unable to keep on living if she could not kiss him that very minute .
[S4] she did so , but by mistake she let three hot drops of tallow fall on him , and he awoke .
[S5] " alas , what have you done ! " cried he .
[S6] " now you have made both of us unhappy . "
[S7] " if you had only held out until the end of the year , i would have been delivered . "
[S8] " i have a step - mother who has cast a spell on me , so that by day i am a bear , and at night a human being . "
[S9] " but now all is over between us , and i must return to my step - mother . "
[S10] " she lives in a castle that is east of the sun and west of the moon , where there is a princess with a nose three yards long , whom i must now marry . "
[S11] the maiden wept and wailed to no avail , for the prince said he must journey away .
[S12] then she asked him whether she might not go with him .
[S13] no , said he , that could not be .
[S14] " but can you not at least tell me the road , so that i can search for you . "
[S15] " for surely that will be permitted me ? "
[S16] " yes , that you may do , " said he .
[S17] " but there is no road that leads there . "
[S18] " the castle lies east of the sun and west of the moon , and neither now nor at any other time will you find the road to it ! "
[S19] when the maiden awoke the next morning , the prince as well as the castle had disappeared .
[S20] she lay in a green opening in the midst of a thick , dark wood , and beside her lay the bundle of poor belongings she had brought from home .
[S21] and when she had rubbed the sleep out of her eyes , and had cried her fill , she set out and wandered many , many days , until at last she came to a great hill .
[S22] and before the hill sat an old woman who was playing with a golden apple .
[S23] the maiden asked the woman whether she did not know which road led to the prince who lived in the castle that was east of the sun and west of the moon , and who was to marry a princess with a nose three yards long .
[S24] " how do you come to know him ? " asked the woman .
[S25] " are you , perhaps , the maiden he wanted to marry ? "
[S26] " yes , i am that maiden , " she replied .

## Sample 70

**story_name:** king-kojata  
**question:** why did hyacinthia go in disguise to the wedding ?  
**answer:** she did not want to be recognized .  

**old sentence count:** 15  
**new sentence count:** 13  

### Old Split

[S0] all night the old man never closed an eye .
[S1] when the first ray of light entered the room , he noticed that the little blue flower began to tremble , and at last it rose out of the pot and flew about the room , put everything in order , swept away the dust , and lit the fire .
[S2] in great haste the old man sprang from his bed , and covered the flower with the cloth the old witch had given him , and in a moment the beautiful princess hyacinthia stood before him .
[S3] ' what have you done ?
[S4] ' she cried .
[S5] ' why have you called me back to life ?
[S6] for i have no desire to live since my bridegroom , the beautiful prince milan , has deserted me .
[S7] ' ' prince milan is just going to be married , ' replied the old man .
[S8] ' everything is being got ready for the feast , and all the invited guests are flocking to the palace from all sides .
[S9] ' the beautiful hyacinthia cried bitterly when she heard this ; then she dried her tears , and went into the town dressed as a peasant woman .
[S10] she went straight to the king 's kitchen , where the white - aproned cooks were running about in great confusion .
[S11] the princess went up to the head cook , and said , ' dear cook , please listen to my request , and let me make a wedding - cake for prince milan .
[S12] ' the busy cook was just going to refuse her demand and order her out of the kitchen , but the words died on his lips when he turned and beheld the beautiful hyacinthia , and he answered politely , ' you have just come in the nick of time , fair maiden .
[S13] bake your cake , and i myself will lay it before prince milan .
[S14] '

### New Split

[S0] all night the old man never closed an eye .
[S1] when the first ray of light entered the room , he noticed that the little blue flower began to tremble , and at last it rose out of the pot and flew about the room , put everything in order , swept away the dust , and lit the fire .
[S2] in great haste the old man sprang from his bed , and covered the flower with the cloth the old witch had given him , and in a moment the beautiful princess hyacinthia stood before him .
[S3] ' what have you done ? ' she cried .
[S4] ' why have you called me back to life ? '
[S5] ' for i have no desire to live since my bridegroom , the beautiful prince milan , has deserted me . '
[S6] ' prince milan is just going to be married , ' replied the old man .
[S7] ' everything is being got ready for the feast , and all the invited guests are flocking to the palace from all sides . '
[S8] the beautiful hyacinthia cried bitterly when she heard this ; then she dried her tears , and went into the town dressed as a peasant woman .
[S9] she went straight to the king 's kitchen , where the white - aproned cooks were running about in great confusion .
[S10] the princess went up to the head cook , and said , ' dear cook , please listen to my request , and let me make a wedding - cake for prince milan . '
[S11] the busy cook was just going to refuse her demand and order her out of the kitchen , but the words died on his lips when he turned and beheld the beautiful hyacinthia , and he answered politely , ' you have just come in the nick of time , fair maiden . '
[S12] ' bake your cake , and i myself will lay it before prince milan . '

## Sample 71

**story_name:** quarrel-of-monkey-and-crab  
**question:** why did the monkey throw persimmons at the crab ?  
**answer:** he was annoyed with the crab .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] you can imagine the feelings of the poor crab after waiting patiently , for so long as he had done , for the tree to grow and the fruit to ripen , when he saw the monkey devouring all the good persimmons .
[S1] he was so disappointed that he ran round and round the tree calling to the monkey to remember his promise .
[S2] the monkey at first took no notice of the crab 's complaints , but at last he picked out the hardest , greenest persimmon he could find and aimed it at the crab 's head .
[S3] the persimmon is as hard as stone when it is unripe .
[S4] the monkey 's missile struck home and the crab was sorely hurt by the blow .
[S5] again and again , as fast as he could pick them , the monkey pulled off the hard persimmons and threw them at the defenseless crab till he dropped dead , covered with wounds all over his body .
[S6] there he lay a pitiful sight at the foot of the tree he had himself planted .

### New Split

[S0] you can imagine the feelings of the poor crab after waiting patiently , for so long as he had done , for the tree to grow and the fruit to ripen , when he saw the monkey devouring all the good persimmons .
[S1] he was so disappointed that he ran round and round the tree calling to the monkey to remember his promise .
[S2] the monkey at first took no notice of the crab 's complaints , but at last he picked out the hardest , greenest persimmon he could find and aimed it at the crab 's head .
[S3] the persimmon is as hard as stone when it is unripe .
[S4] the monkey 's missile struck home and the crab was sorely hurt by the blow .
[S5] again and again , as fast as he could pick them , the monkey pulled off the hard persimmons and threw them at the defenseless crab till he dropped dead , covered with wounds all over his body .
[S6] there he lay a pitiful sight at the foot of the tree he had himself planted .

## Sample 72

**story_name:** the-brown-bear-of-norway  
**question:** why did the youngest princess's husband disappear when she woke up ?  
**answer:** he turned into a bear during the day .  

**old sentence count:** 13  
**new sentence count:** 13  

### Old Split

[S0] but that very night she woke up out of her sleep in a great hall that was lighted up with a thousand lamps ; the richest carpets were on the floor , and the walls were covered with cloth of gold and silver , and the place was full of grand company , and the very beautiful prince she saw in her dreams was there , and it was n't a moment till he was on one knee before her , and telling her how much he loved her , and asking her would n't she be his queen .
[S1] well , she had n't the heart to refuse him , and married they were the same evening .
[S2] ' now , my darling , ' says he , when they were left by themselves , ' you must know that i am under enchantment .
[S3] a sorceress , that had a beautiful daughter , wished me for her son - in - law ; but the mother got power over me , and when i refused to we d her daughter she made me take the form of a bear by day , and i was to continue so till a lady would marry me of her own free will , and endure five years of great trials after .
[S4] ' well , when the princess woke in the morning , she missed her husband from her side , and spent the day very sadly .
[S5] but as soon as the lamps were lighted in the grand hall , where she was sitting on a sofa covered with silk , the folding doors flew open , and he was sitting by her side the next minute .
[S6] so they spent another happy evening , but he warned her that whenever she began to tire of him , or ceased to have faith in him , they would be parted for ever , and he 'd be obliged to marry the witch 's daughter .
[S7] she got used to find him absent by day , and they spent a happy twelvemonth together , and at last a beautiful little boy was born ; and happy as she was before , she was twice as happy now , for she had her child to keep her company in the day when she could n't see her husband .
[S8] at last , one evening , when herself , and himself , and her child were sitting with a window open because it was a sultry night , in flew an eagle , took the infant 's sash in his beak , and flew up in the air with him .
[S9] she screamed , and was going to throw herself out the window after him , but the prince caught her , and looked at her very seriously .
[S10] she bethought of what he said soon after their marriage , and she stopped the cries and complaints that were on her tongue .
[S11] she spent her days very lonely for another twelvemonth , when a beautiful little girl was sent to her .
[S12] then she thought to herself she 'd have a sharp eye about her this time ; so she never would allow a window to be more than a few inches open .

### New Split

[S0] but that very night she woke up out of her sleep in a great hall that was lighted up with a thousand lamps ; the richest carpets were on the floor , and the walls were covered with cloth of gold and silver , and the place was full of grand company , and the very beautiful prince she saw in her dreams was there , and it was n't a moment till he was on one knee before her , and telling her how much he loved her , and asking her would n't she be his queen .
[S1] well , she had n't the heart to refuse him , and married they were the same evening .
[S2] ' now , my darling , ' says he , when they were left by themselves , ' you must know that i am under enchantment . '
[S3] ' a sorceress , that had a beautiful daughter , wished me for her son - in - law ; but the mother got power over me , and when i refused to we d her daughter she made me take the form of a bear by day , and i was to continue so till a lady would marry me of her own free will , and endure five years of great trials after . '
[S4] well , when the princess woke in the morning , she missed her husband from her side , and spent the day very sadly .
[S5] but as soon as the lamps were lighted in the grand hall , where she was sitting on a sofa covered with silk , the folding doors flew open , and he was sitting by her side the next minute .
[S6] so they spent another happy evening , but he warned her that whenever she began to tire of him , or ceased to have faith in him , they would be parted for ever , and he 'd be obliged to marry the witch 's daughter .
[S7] she got used to find him absent by day , and they spent a happy twelvemonth together , and at last a beautiful little boy was born ; and happy as she was before , she was twice as happy now , for she had her child to keep her company in the day when she could n't see her husband .
[S8] at last , one evening , when herself , and himself , and her child were sitting with a window open because it was a sultry night , in flew an eagle , took the infant 's sash in his beak , and flew up in the air with him .
[S9] she screamed , and was going to throw herself out the window after him , but the prince caught her , and looked at her very seriously .
[S10] she bethought of what he said soon after their marriage , and she stopped the cries and complaints that were on her tongue .
[S11] she spent her days very lonely for another twelvemonth , when a beautiful little girl was sent to her .
[S12] then she thought to herself she 'd have a sharp eye about her this time ; so she never would allow a window to be more than a few inches open .

## Sample 73

**story_name:** anent-giant-who-did-not-have-his-heart-about-him  
**question:** how was the raven able to retrieve the key ?  
**answer:** he could fly .  

**old sentence count:** 18  
**new sentence count:** 16  

### Old Split

[S0] " you must call the raven , " said the wolf , and that is what the king 's son did .
[S1] and the raven came at once , and flew right down with the key , and now the prince could enter the church .
[S2] then , when he came to the well , there was the duck , sure enough , swimming about as the giant had said .
[S3] he stood by the well and called the duck , and at last he lured her near him , and seized her .
[S4] but at the moment he grasped her and lifted her out of the water , she let the egg fall into the well .
[S5] now the prince again did not know how he was to get hold of it .
[S6] " well , you must call the salmon , " said the wolf .
[S7] that is what the king 's son did , and the salmon came at once , and brought up the egg from the bottom of the well .
[S8] then the wolf told him to squeeze the egg a little .
[S9] and when the prince squeezed , the giant cried out .
[S10] " squeeze it again !
[S11] " said the wolf , and when the prince did so , the giant cried out far more dolefully , and fearfully and tearfully begged for his life .
[S12] he would do all the king 's son asked him to , said he , if only he would not squeeze his heart in two .
[S13] " tell him to give back their original form to your six brothers , whom he turned to stone , and to their brides , as well ; and that then you will spare his life , " said the wolf , and the prince did so .
[S14] the troll at once agreed , and changed the six brothers into princes , and their brides into kings ' daughters .
[S15] " now squash the egg !
[S16] " cried the wolf .
[S17] then the prince squeezed the egg in two , and the giant burst into pieces .

### New Split

[S0] " you must call the raven , " said the wolf , and that is what the king 's son did .
[S1] and the raven came at once , and flew right down with the key , and now the prince could enter the church .
[S2] then , when he came to the well , there was the duck , sure enough , swimming about as the giant had said .
[S3] he stood by the well and called the duck , and at last he lured her near him , and seized her .
[S4] but at the moment he grasped her and lifted her out of the water , she let the egg fall into the well .
[S5] now the prince again did not know how he was to get hold of it .
[S6] " well , you must call the salmon , " said the wolf .
[S7] that is what the king 's son did , and the salmon came at once , and brought up the egg from the bottom of the well .
[S8] then the wolf told him to squeeze the egg a little .
[S9] and when the prince squeezed , the giant cried out .
[S10] " squeeze it again ! " said the wolf , and when the prince did so , the giant cried out far more dolefully , and fearfully and tearfully begged for his life .
[S11] he would do all the king 's son asked him to , said he , if only he would not squeeze his heart in two .
[S12] " tell him to give back their original form to your six brothers , whom he turned to stone , and to their brides , as well ; and that then you will spare his life , " said the wolf , and the prince did so .
[S13] the troll at once agreed , and changed the six brothers into princes , and their brides into kings ' daughters .
[S14] " now squash the egg ! " cried the wolf .
[S15] then the prince squeezed the egg in two , and the giant burst into pieces .

## Sample 74

**story_name:** story-of-old-man-who-made-withered-trees-to-flower  
**question:** how did the old man feel finding all the coins ?  
**answer:** excited .  

**old sentence count:** 8  
**new sentence count:** 7  

### Old Split

[S0] the thought that something might be hidden beneath the tree , and that the dog had scented it , at last struck the old man .
[S1] he ran back to the house , fetched his spade and began to dig the ground at that spot .
[S2] what was his astonishment when , after digging for some time , he came upon a heap of old and valuable coins .
[S3] the deeper he dug the more gold coins did he find .
[S4] so intent was the old man on his work that he never saw the cross face of his neighbor peering at him through the bamboo hedge .
[S5] at last all the gold coins lay shining on the ground .
[S6] shiro sat by erect with pride and looking fondly at his master as if to say , " you see , though only a dog , i can make some return for all the kindness you show me .
[S7] "

### New Split

[S0] the thought that something might be hidden beneath the tree , and that the dog had scented it , at last struck the old man .
[S1] he ran back to the house , fetched his spade and began to dig the ground at that spot .
[S2] what was his astonishment when , after digging for some time , he came upon a heap of old and valuable coins .
[S3] the deeper he dug the more gold coins did he find .
[S4] so intent was the old man on his work that he never saw the cross face of his neighbor peering at him through the bamboo hedge .
[S5] at last all the gold coins lay shining on the ground .
[S6] shiro sat by erect with pride and looking fondly at his master as if to say , " you see , though only a dog , i can make some return for all the kindness you show me . "

## Sample 75

**story_name:** the-red-swan  
**question:** why did maidwa walk towards the lodges of a large village ?  
**answer:** it was night .  

**old sentence count:** 21  
**new sentence count:** 20  

### Old Split

[S0] " the bird is mine , " said maidwa , to himself .
[S1] but to his great surprise , instead of seeing it droop its neck and drift to the shore , the red swan flapped its wings , rose slowly , and flew off with a majestic motion toward the falling sun .
[S2] maidwa , that he might meet his brothers , rescued two of the magic arrows from the water ; and although the third was borne off , he had a hope yet to recover that too , and to be master of the swan .
[S3] he was noted for his speed ; for he would shoot an arrow and then run so fast that the arrow always fell behind him ; and he now set off at his best speed of foot .
[S4] " i can run fast , " he thought , " and i can get up with the swan some time or other .
[S5] " he sped on , over hills and prairies , toward the west , and was only going to take one more run , and then seek a place to sleep for the night , when , suddenly , he heard noises at a distance , like the murmur of waters against the shore .
[S6] as he went on , he heard voices , and presently he saw people , some of whom were busy felling trees , and the strokes of their labor echoed through the woods .
[S7] he passed on , and when he emerged from the forest , the sun was just falling below the edge of the sky .
[S8] he was bent on success in pursuit of the swan , whose red track he marked well far westward till she was lost to sight .
[S9] meanwhile he would tarry for the night and procure something to eat , as he had fasted since he had left home .
[S10] at a distance , on a rising ground , he could see the lodges of a large village .
[S11] he went toward it , and soon heard the watchman , who was set on a height to overlook the place , and give notice of the approach of friends or foes , crying out , " we are visited ; " and a loud halloo indicated that they had all heard it .
[S12] when maidwa advanced , the watchman pointed to the lodge of the chief .
[S13] " it is there you must go in , " he said , and left him .
[S14] " come in , come in , " said the chief ; " take a seat there ; " pointing to the side of the lodge where his daughter sat .
[S15] " it is there you must sit .
[S16] " they gave him something to eat , and , being a stranger , very few questions were put to him .
[S17] it was only when he spoke that the others answered him .
[S18] " daughter , " said the chief , as soon as the night had set in , " take our son - in - law 's moccasins and see if they be torn .
[S19] if so , mend them for him , and bring in his bundle .
[S20] "

### New Split

[S0] " the bird is mine , " said maidwa , to himself .
[S1] but to his great surprise , instead of seeing it droop its neck and drift to the shore , the red swan flapped its wings , rose slowly , and flew off with a majestic motion toward the falling sun .
[S2] maidwa , that he might meet his brothers , rescued two of the magic arrows from the water ; and although the third was borne off , he had a hope yet to recover that too , and to be master of the swan .
[S3] he was noted for his speed ; for he would shoot an arrow and then run so fast that the arrow always fell behind him ; and he now set off at his best speed of foot .
[S4] " i can run fast , " he thought , " and i can get up with the swan some time or other . "
[S5] he sped on , over hills and prairies , toward the west , and was only going to take one more run , and then seek a place to sleep for the night , when , suddenly , he heard noises at a distance , like the murmur of waters against the shore .
[S6] as he went on , he heard voices , and presently he saw people , some of whom were busy felling trees , and the strokes of their labor echoed through the woods .
[S7] he passed on , and when he emerged from the forest , the sun was just falling below the edge of the sky .
[S8] he was bent on success in pursuit of the swan , whose red track he marked well far westward till she was lost to sight .
[S9] meanwhile he would tarry for the night and procure something to eat , as he had fasted since he had left home .
[S10] at a distance , on a rising ground , he could see the lodges of a large village .
[S11] he went toward it , and soon heard the watchman , who was set on a height to overlook the place , and give notice of the approach of friends or foes , crying out , " we are visited ; " and a loud halloo indicated that they had all heard it .
[S12] when maidwa advanced , the watchman pointed to the lodge of the chief .
[S13] " it is there you must go in , " he said , and left him .
[S14] " come in , come in , " said the chief ; " take a seat there ; " pointing to the side of the lodge where his daughter sat .
[S15] " it is there you must sit . "
[S16] they gave him something to eat , and , being a stranger , very few questions were put to him .
[S17] it was only when he spoke that the others answered him .
[S18] " daughter , " said the chief , as soon as the night had set in , " take our son - in - law 's moccasins and see if they be torn . "
[S19] " if so , mend them for him , and bring in his bundle . "

## Sample 76

**story_name:** the-queen-of-heaven  
**question:** why were there three paper talisman and a picture of the queen of heaven were kept on the ship ?  
**answer:** the seamen asked for help when there were in danger .  

**old sentence count:** 5  
**new sentence count:** 5  

### Old Split

[S0] in every ship that sails a picture of the queen of heaven hangs in the cabin , and three paper talismans are also kept on shipboard .
[S1] on the first she is painted with crown and scepter , on the second as a maiden in ordinary dress , and on the third she is pictured with flowing hair , barefoot , standing with a sword in her hand .
[S2] when the ship is in danger the first talisman is burnt , and help comes .
[S3] but if this is of no avail , then the second and finally the third picture is burned .
[S4] and if no help comes then there is nothing more to be done .

### New Split

[S0] in every ship that sails a picture of the queen of heaven hangs in the cabin , and three paper talismans are also kept on shipboard .
[S1] on the first she is painted with crown and scepter , on the second as a maiden in ordinary dress , and on the third she is pictured with flowing hair , barefoot , standing with a sword in her hand .
[S2] when the ship is in danger the first talisman is burnt , and help comes .
[S3] but if this is of no avail , then the second and finally the third picture is burned .
[S4] and if no help comes then there is nothing more to be done .

## Sample 77

**story_name:** goblin-huckster  
**question:** why did the goblin live with the huckster ?  
**answer:** the huckster had jam at christmas .  

**old sentence count:** 4  
**new sentence count:** 4  

### Old Split

[S0] there was once a regular student , who lived in a garret and had no possessions .
[S1] and there was also a regular huckster , to whom the house belonged , and who occupied the ground floor .
[S2] a goblin lived with the huckster because at christmas he always had a large dishful of jam , with a great piece of butter in the middle .
[S3] the huckster could afford this , and therefore the goblin remained with him -- which was very shrewd of the goblin .

### New Split

[S0] there was once a regular student , who lived in a garret and had no possessions .
[S1] and there was also a regular huckster , to whom the house belonged , and who occupied the ground floor .
[S2] a goblin lived with the huckster because at christmas he always had a large dishful of jam , with a great piece of butter in the middle .
[S3] the huckster could afford this , and therefore the goblin remained with him -- which was very shrewd of the goblin .

## Sample 78

**story_name:** the-one-handed-girl  
**question:** why did the girl cry in the forest ?  
**answer:** she faced a lot of obstacles after her parents ' deaths .  

**old sentence count:** 14  
**new sentence count:** 11  

### Old Split

[S0] ' what is the matter with you ?
[S1] ' said he gently , and , as she only sobbed louder , he continued : ' are you a woman , or a spirit of the woods ?
[S2] ' ' i am a woman , ' she answered slowly , wiping her eyes with a leaf of the creeper that hung about her .
[S3] ' then why do you cry ?
[S4] ' he persisted .
[S5] ' i have many things to cry for , ' she replied , ' more than you could ever guess .
[S6] ' ' come home with me , ' said the prince ; ' it is not very far .
[S7] come home to my father and mother .
[S8] i am a king 's son .
[S9] ' ' then why are you here ?
[S10] ' she said , opening her eyes and staring at him .
[S11] ' once every month i and my friends shoot birds in the forest , ' he answered , ' but i was tired and bade them leave me to rest .
[S12] and you -- what are you doing up in this tree ?
[S13] '

### New Split

[S0] ' what is the matter with you ? ' said he gently , and , as she only sobbed louder , he continued : ' are you a woman , or a spirit of the woods ? '
[S1] ' i am a woman , ' she answered slowly , wiping her eyes with a leaf of the creeper that hung about her .
[S2] ' then why do you cry ? '
[S3] he persisted .
[S4] ' i have many things to cry for , ' she replied , ' more than you could ever guess . '
[S5] ' come home with me , ' said the prince ; ' it is not very far . '
[S6] ' come home to my father and mother . '
[S7] ' i am a king 's son . '
[S8] ' then why are you here ? ' she said , opening her eyes and staring at him .
[S9] ' once every month i and my friends shoot birds in the forest , ' he answered , ' but i was tired and bade them leave me to rest . '
[S10] ' and you -- what are you doing up in this tree ? '

## Sample 79

**story_name:** east-of-sun-and-west-of-moon  
**question:** why had the prince been asleep when the princess came in ?  
**answer:** the princess had been tricking him .  

**old sentence count:** 17  
**new sentence count:** 17  

### Old Split

[S0] that day the maiden again sat beneath the windows of the castle , and wound her golden reel .
[S1] and all went as on the preceding day .
[S2] the princess asked what she wanted for the reel , and the maiden answered that she would sell it neither for gold nor for money .
[S3] if she might speak that night to the prince , then she would give the reel to the princess .
[S4] yet when the maiden came to the prince , he was again fast asleep , and no matter how much she wept and wailed , and cried and shook , she could not wake him .
[S5] but as soon as day dawned , and it grew bright , the princess with the long nose came and drove her out .
[S6] and that day the maiden again seated herself beneath the windows of the castle , and spun with her golden spindle .
[S7] of course , the princess with the long nose wanted to have that , too .
[S8] she opened the window , and asked what she wanted for the golden spindle .
[S9] the maiden replied , as she had twice before , that she would sell the spindle neither for gold nor money .
[S10] the princess could have it if she might speak to the prince again that night .
[S11] yes , that she was welcome to do , said the princess , and took the golden spindle .
[S12] now it happened that some christians , who were captives in the castle , and quartered in a room beside that of the prince , had heard a woman weeping and wailing pitifully in the prince 's room for the past two nights .
[S13] so they told the prince .
[S14] and that evening when the princess came to him with his night - cap , the prince pretended to drink it .
[S15] he instead poured it out behind his back , for he could well imagine that she had put a sleeping - powder into the cup .
[S16] then , when the maiden came in , the prince was awake , and she had to tell him just how she had found the castle .

### New Split

[S0] that day the maiden again sat beneath the windows of the castle , and wound her golden reel .
[S1] and all went as on the preceding day .
[S2] the princess asked what she wanted for the reel , and the maiden answered that she would sell it neither for gold nor for money .
[S3] if she might speak that night to the prince , then she would give the reel to the princess .
[S4] yet when the maiden came to the prince , he was again fast asleep , and no matter how much she wept and wailed , and cried and shook , she could not wake him .
[S5] but as soon as day dawned , and it grew bright , the princess with the long nose came and drove her out .
[S6] and that day the maiden again seated herself beneath the windows of the castle , and spun with her golden spindle .
[S7] of course , the princess with the long nose wanted to have that , too .
[S8] she opened the window , and asked what she wanted for the golden spindle .
[S9] the maiden replied , as she had twice before , that she would sell the spindle neither for gold nor money .
[S10] the princess could have it if she might speak to the prince again that night .
[S11] yes , that she was welcome to do , said the princess , and took the golden spindle .
[S12] now it happened that some christians , who were captives in the castle , and quartered in a room beside that of the prince , had heard a woman weeping and wailing pitifully in the prince 's room for the past two nights .
[S13] so they told the prince .
[S14] and that evening when the princess came to him with his night - cap , the prince pretended to drink it .
[S15] he instead poured it out behind his back , for he could well imagine that she had put a sleeping - powder into the cup .
[S16] then , when the maiden came in , the prince was awake , and she had to tell him just how she had found the castle .

## Sample 80

**story_name:** the-believing-husbands  
**question:** why were the two men scared after they heard the man from the coffin talk ?  
**answer:** the man from the coffin was alive .  

**old sentence count:** 11  
**new sentence count:** 9  

### Old Split

[S0] ' do you know me ?
[S1] ' ' not i , ' answered the naked man .
[S2] ' i do not know you .
[S3] ' ' but why are you naked ?
[S4] ' asked the first man .
[S5] ' am i naked ?
[S6] my wife told me that i had all my clothes on , ' answered he .
[S7] ' and my wife told me that i myself was dead , ' said the man in the coffin .
[S8] but at the sound of his voice the two men were so terrified that they ran straight home .
[S9] the man in the coffin got up and followed them .
[S10] it was his wife that gained the gold ring , as he had been sillier than the other two .

### New Split

[S0] ' do you know me ? '
[S1] ' not i , ' answered the naked man .
[S2] ' i do not know you . '
[S3] ' but why are you naked ? ' asked the first man .
[S4] ' am i naked ? my wife told me that i had all my clothes on , ' answered he .
[S5] ' and my wife told me that i myself was dead , ' said the man in the coffin .
[S6] but at the sound of his voice the two men were so terrified that they ran straight home .
[S7] the man in the coffin got up and followed them .
[S8] it was his wife that gained the gold ring , as he had been sillier than the other two .

## Sample 81

**story_name:** the-fox-and-the-wolf  
**question:** why was the fox worried about going to the christening ?  
**answer:** he must travel far .  

**old sentence count:** 15  
**new sentence count:** 11  

### Old Split

[S0] about a week passed by : then one day the fox came into the cave , and flung himself down on the ground as if he were very much exhausted .
[S1] but if anyone had looked at him closely they would have seen a sly twinkle in his eye .
[S2] " oh , dear , oh , dear !
[S3] " he sighed .
[S4] " life is a heavy burden .
[S5] " " what have befallen you ?
[S6] " asked the wolf , who was ever kind and soft - hearted .
[S7] " some friends of mine , who live over the hills yonder , are wanting me to go to a christening to - night .
[S8] just think of the distance that i must travel .
[S9] " " but do you need to go ?
[S10] " asked the wolf .
[S11] " can you not send an excuse ?
[S12] " " i doubt that no excuse would be accepted , " answered the fox , " for they asked me to stand god - father .
[S13] therefore it behoveth me to do my duty , and pay no heed to my own feelings .
[S14] "

### New Split

[S0] about a week passed by : then one day the fox came into the cave , and flung himself down on the ground as if he were very much exhausted .
[S1] but if anyone had looked at him closely they would have seen a sly twinkle in his eye .
[S2] " oh , dear , oh , dear ! " he sighed .
[S3] " life is a heavy burden . "
[S4] " what have befallen you ? " asked the wolf , who was ever kind and soft - hearted .
[S5] " some friends of mine , who live over the hills yonder , are wanting me to go to a christening to - night . "
[S6] " just think of the distance that i must travel . "
[S7] " but do you need to go ? " asked the wolf .
[S8] " can you not send an excuse ? "
[S9] " i doubt that no excuse would be accepted , " answered the fox , " for they asked me to stand god - father . "
[S10] " therefore it behoveth me to do my duty , and pay no heed to my own feelings . "

## Sample 82

**story_name:** the-wolf-and-the-seven-little-goats  
**question:** how did the wolf threaten the miller ?  
**answer:** he threatened to eat him up .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] the wolf then ran to a baker .
[S1] " baker , " said he , " i am hurt in the foot ; pray spread some dough over the place .
[S2] " and when the baker had plastered his feet , he ran to the miller .
[S3] " miller , " said he , " strew me some white meal over my paws .
[S4] " but the miller refused , thinking the wolf must be meaning harm to some one .
[S5] " if you do n't do it , " cried the wolf , " i 'll eat you up !
[S6] " and the miller was afraid and did as he was told .
[S7] and that just shows what men are .

### New Split

[S0] the wolf then ran to a baker .
[S1] " baker , " said he , " i am hurt in the foot ; pray spread some dough over the place . "
[S2] and when the baker had plastered his feet , he ran to the miller .
[S3] " miller , " said he , " strew me some white meal over my paws . "
[S4] but the miller refused , thinking the wolf must be meaning harm to some one .
[S5] " if you do n't do it , " cried the wolf , " i 'll eat you up ! "
[S6] and the miller was afraid and did as he was told .
[S7] and that just shows what men are .

## Sample 83

**story_name:** habetrot-the-spinstress  
**question:** why did the rich young nobleman ask to see maisie ?  
**answer:** he was impressed by her spinning skills .  

**old sentence count:** 4  
**new sentence count:** 4  

### Old Split

[S0] he stopped his horse , and said good - naturedly , " good day , madam ; and may i ask why you sing such a strange song ?
[S1] " maisie 's mother made no answer , but turned and walked into the house ; and the young nobleman , being very anxious to know what it all meant , hung his bridle over the garden gate , and followed her .
[S2] she pointed to the seven hanks of thread lying on the table , and said , " this hath my daughter done before breakfast .
[S3] " then the young man asked to see the maiden who was so industrious , and her mother went and pulled maisie from behind the door , where she had hidden herself when the stranger came in ; for she had come downstairs while her mother was in the garden .

### New Split

[S0] he stopped his horse , and said good - naturedly , " good day , madam ; and may i ask why you sing such a strange song ? "
[S1] maisie 's mother made no answer , but turned and walked into the house ; and the young nobleman , being very anxious to know what it all meant , hung his bridle over the garden gate , and followed her .
[S2] she pointed to the seven hanks of thread lying on the table , and said , " this hath my daughter done before breakfast . "
[S3] then the young man asked to see the maiden who was so industrious , and her mother went and pulled maisie from behind the door , where she had hidden herself when the stranger came in ; for she had come downstairs while her mother was in the garden .

## Sample 84

**story_name:** the-fairy-nurse  
**question:** who did the neighbor see ?  
**answer:** the poor man 's wife .  

**old sentence count:** 7  
**new sentence count:** 6  

### Old Split

[S0] there they stood , looking towards the bridge of thuar , in the dead of the night , with a little moonlight shining from over kilachdiarmid .
[S1] at last she gave a start , and " by this and by that , " says she , " here they come , bridles jingling and feathers tossing !
[S2] " he looked , but could see nothing ; and she stood trembling and her eyes wide open , looking down the way to the ford of ballinacoola .
[S3] " i see your wife , " says she , " riding on the outside just so as to rub against us .
[S4] we 'll walk on quietly , as if we suspected nothing , and when we are passing i 'll give you a shove .
[S5] if you do n't do your duty then , woe be with you !
[S6] "

### New Split

[S0] there they stood , looking towards the bridge of thuar , in the dead of the night , with a little moonlight shining from over kilachdiarmid .
[S1] at last she gave a start , and " by this and by that , " says she , " here they come , bridles jingling and feathers tossing ! "
[S2] he looked , but could see nothing ; and she stood trembling and her eyes wide open , looking down the way to the ford of ballinacoola .
[S3] " i see your wife , " says she , " riding on the outside just so as to rub against us . "
[S4] " we 'll walk on quietly , as if we suspected nothing , and when we are passing i 'll give you a shove . "
[S5] " if you do n't do your duty then , woe be with you ! "

## Sample 85

**story_name:** hans-in-luck  
**question:** why did hans trade his horse for the cow ?  
**answer:** because the horse was difficult to ride .  

**old sentence count:** 19  
**new sentence count:** 18  

### Old Split

[S0] and hans , as he sat upon his horse , was glad at heart , and rode off with merry cheer .
[S1] after a while he thought he should like to go quicker , so he began to click with his tongue and to cry " gee - up !
[S2] " and the horse began to trot , and hans was thrown before he knew what was going to happen .
[S3] there he lay in the ditch by the side of the road .
[S4] the horse would have got away but that he was caught by a peasant who was passing that way and driving a cow before him .
[S5] and hans pulled himself together and got upon his feet , feeling very vexed .
[S6] " poor work , riding , " said he .
[S7] " especially on a jade like this , who starts off and throws you before you know where you are , going near to break your neck .
[S8] never shall i try that game again .
[S9] now , your cow is something worth having .
[S10] one can jog on comfortably after her and have her milk , butter , and cheese every day , into the bargain .
[S11] what would i not give to have such a cow !
[S12] " " well now , " said the peasant , " since it will be doing you such a favour , i do n't mind exchanging my cow for your horse .
[S13] " hans agreed most joyfully , and the peasant , swinging himself into the saddle , was soon out of sight .
[S14] and hans went along driving his cow quietly before him , and thinking all the while of the fine bargain he had made .
[S15] " with only a piece of bread i shall have everything i can possibly want , for i shall always be able to have butter and cheese to it .
[S16] if i am thirsty i have nothing to do but to milk my cow .
[S17] what more is there for heart to wish !
[S18] "

### New Split

[S0] and hans , as he sat upon his horse , was glad at heart , and rode off with merry cheer .
[S1] after a while he thought he should like to go quicker , so he began to click with his tongue and to cry " gee - up ! "
[S2] and the horse began to trot , and hans was thrown before he knew what was going to happen .
[S3] there he lay in the ditch by the side of the road .
[S4] the horse would have got away but that he was caught by a peasant who was passing that way and driving a cow before him .
[S5] and hans pulled himself together and got upon his feet , feeling very vexed .
[S6] " poor work , riding , " said he .
[S7] " especially on a jade like this , who starts off and throws you before you know where you are , going near to break your neck . "
[S8] " never shall i try that game again . "
[S9] " now , your cow is something worth having . "
[S10] " one can jog on comfortably after her and have her milk , butter , and cheese every day , into the bargain . "
[S11] " what would i not give to have such a cow ! "
[S12] " well now , " said the peasant , " since it will be doing you such a favour , i do n't mind exchanging my cow for your horse . "
[S13] hans agreed most joyfully , and the peasant , swinging himself into the saddle , was soon out of sight .
[S14] and hans went along driving his cow quietly before him , and thinking all the while of the fine bargain he had made .
[S15] " with only a piece of bread i shall have everything i can possibly want , for i shall always be able to have butter and cheese to it . "
[S16] " if i am thirsty i have nothing to do but to milk my cow . "
[S17] " what more is there for heart to wish ! "

## Sample 86

**story_name:** the-rich-brother-and-the-poor-brother  
**question:** what will happen after the husband is not able to get the unfinished houses back ?  
**answer:** his wife will keep trying to fight for the unfinished houses .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] at this answer the wife grew very angry .
[S1] she began to cry , and made such a noise that all the neighbours heard her and put their heads out of the windows , to see what was the matter .
[S2] ' it was absurd , ' she sobbed out , ' quite unjust .
[S3] indeed , if you came to think of it , the gift was worth nothing , as when her husband made it he was a bachelor , and since then he had been married , and she had never given her consent to any such thing .
[S4] ' and so she lamented all day and all night , till the poor man was nearly worried to death ; and at last he did what she wished , and summoned his brother in a court of law to give up the houses which , he said , had only been lent to him .
[S5] but when the evidence on both sides had been heard , the judge decided in favour of the poor man , which made the rich lady more furious than ever , and she determined not to rest until she had gained the day .
[S6] if one judge would not give her the houses another should , and so time after time the case was tried over again , till at last it came before the highest judge of all , in the city of evora .
[S7] her husband was heartily tired and ashamed of the whole affair , but his weakness in not putting a stop to it in the beginning had got him into this difficulty , and now he was forced to go on .

### New Split

[S0] at this answer the wife grew very angry .
[S1] she began to cry , and made such a noise that all the neighbours heard her and put their heads out of the windows , to see what was the matter .
[S2] ' it was absurd , ' she sobbed out , ' quite unjust . '
[S3] ' indeed , if you came to think of it , the gift was worth nothing , as when her husband made it he was a bachelor , and since then he had been married , and she had never given her consent to any such thing . '
[S4] and so she lamented all day and all night , till the poor man was nearly worried to death ; and at last he did what she wished , and summoned his brother in a court of law to give up the houses which , he said , had only been lent to him .
[S5] but when the evidence on both sides had been heard , the judge decided in favour of the poor man , which made the rich lady more furious than ever , and she determined not to rest until she had gained the day .
[S6] if one judge would not give her the houses another should , and so time after time the case was tried over again , till at last it came before the highest judge of all , in the city of evora .
[S7] her husband was heartily tired and ashamed of the whole affair , but his weakness in not putting a stop to it in the beginning had got him into this difficulty , and now he was forced to go on .

## Sample 87

**story_name:** east-of-sun-and-west-of-moon  
**question:** why were both the parents and maiden doing well ?  
**answer:** they both had money .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] " this is where your parents live , " said the white bear .
[S1] " only do not forget what i told you , or you will make us both unhappy .
[S2] " heaven forbid that she should forget it , said the maiden .
[S3] when she had come to the house , she got down , and the bear turned back .
[S4] when the daughter entered her parents ' home , they were more than happy .
[S5] they told her that they could not thank her enough for what she had done , and that now all of them were doing splendidly .
[S6] then they asked her how she herself fared .
[S7] the maiden answered that all was well with her , also , and that she had all that heart could desire .

### New Split

[S0] " this is where your parents live , " said the white bear .
[S1] " only do not forget what i told you , or you will make us both unhappy . "
[S2] heaven forbid that she should forget it , said the maiden .
[S3] when she had come to the house , she got down , and the bear turned back .
[S4] when the daughter entered her parents ' home , they were more than happy .
[S5] they told her that they could not thank her enough for what she had done , and that now all of them were doing splendidly .
[S6] then they asked her how she herself fared .
[S7] the maiden answered that all was well with her , also , and that she had all that heart could desire .

## Sample 88

**story_name:** the-snow-man  
**question:** how will the snow-man feel when he cannot reach the stove before he melts ?  
**answer:** sad .  

**old sentence count:** 39  
**new sentence count:** 35  

### Old Split

[S0] the whole day the snow - man looked through the window ; towards dusk the room grew still more inviting ; the stove gave out a mild light , not at all like the moon or even the sun ; no , as only a stove can shine , when it has something to feed upon .
[S1] when the door of the room was open , it flared up - this was one of its peculiarities ; it flickered quite red upon the snow - man 's white face .
[S2] ' i ca n't stand it any longer !
[S3] ' he said .
[S4] ' how beautiful it looks with its tongue stretched out like that !
[S5] ' it was a long night , but the snow - man did not find it so ; there he stood , wrapt in his pleasant thoughts , and they froze , so that he cracked .
[S6] next morning the panes of the kitchen window were covered with ice , and the most beautiful ice - flowers that even a snow - man could desire , only they blotted out the stove .
[S7] the window would not open ; he could n't see the stove which he thought was such a lovely lady .
[S8] there was a cracking and cracking inside him and all around ; there was just such a frost as a snow - man would delight in .
[S9] but this snow - man was different : how could he feel happy ?
[S10] ' yours is a bad illness for a snow - man !
[S11] ' said the yard - dog .
[S12] ' i also suffered from it , but i have got over it .
[S13] bow - wow !
[S14] ' he barked .
[S15] ' the weather is going to change !
[S16] ' he added .
[S17] the weather did change .
[S18] there came a thaw .
[S19] when this set in the snow - man set off .
[S20] he did not say anything , and he did not complain , and those are bad signs .
[S21] one morning he broke up altogether .
[S22] and lo !
[S23] where he had stood there remained a broomstick standing upright , round which the boys had built him !
[S24] ' ah !
[S25] now i understand why he loved the stove , ' said the yard - dog .
[S26] ' that is the raker they use to clean out the stove !
[S27] the snow - man had a stove - raker in his body !
[S28] that 's what was the matter with him !
[S29] and now it 's all over with him !
[S30] bow - wow !
[S31] ' and before long it was all over with the winter too !
[S32] ' bow - wow !
[S33] ' barked the hoarse yard - dog .
[S34] but the young girl sang : woods , your bright green garments don !
[S35] willows , your woolly gloves put on !
[S36] lark and cuckoo , daily sing- february has brought the spring !
[S37] my heart joins in your song so sweet ; come out , dear sun , the world to greet !
[S38] and no one thought of the snow - man .

### New Split

[S0] the whole day the snow - man looked through the window ; towards dusk the room grew still more inviting ; the stove gave out a mild light , not at all like the moon or even the sun ; no , as only a stove can shine , when it has something to feed upon .
[S1] when the door of the room was open , it flared up - this was one of its peculiarities ; it flickered quite red upon the snow - man 's white face .
[S2] ' i ca n't stand it any longer ! ' he said .
[S3] ' how beautiful it looks with its tongue stretched out like that ! '
[S4] it was a long night , but the snow - man did not find it so ; there he stood , wrapt in his pleasant thoughts , and they froze , so that he cracked .
[S5] next morning the panes of the kitchen window were covered with ice , and the most beautiful ice - flowers that even a snow - man could desire , only they blotted out the stove .
[S6] the window would not open ; he could n't see the stove which he thought was such a lovely lady .
[S7] there was a cracking and cracking inside him and all around ; there was just such a frost as a snow - man would delight in .
[S8] but this snow - man was different : how could he feel happy ?
[S9] ' yours is a bad illness for a snow - man ! ' said the yard - dog .
[S10] ' i also suffered from it , but i have got over it . '
[S11] ' bow - wow ! '
[S12] he barked .
[S13] ' the weather is going to change ! ' he added .
[S14] the weather did change .
[S15] there came a thaw .
[S16] when this set in the snow - man set off .
[S17] he did not say anything , and he did not complain , and those are bad signs .
[S18] one morning he broke up altogether .
[S19] and lo !
[S20] where he had stood there remained a broomstick standing upright , round which the boys had built him !
[S21] ' ah ! now i understand why he loved the stove , ' said the yard - dog .
[S22] ' that is the raker they use to clean out the stove ! '
[S23] ' the snow - man had a stove - raker in his body ! '
[S24] ' that 's what was the matter with him ! '
[S25] ' and now it 's all over with him ! '
[S26] ' bow - wow ! '
[S27] and before long it was all over with the winter too !
[S28] ' bow - wow ! '
[S29] barked the hoarse yard - dog .
[S30] but the young girl sang : woods , your bright green garments don !
[S31] willows , your woolly gloves put on !
[S32] lark and cuckoo , daily sing- february has brought the spring !
[S33] my heart joins in your song so sweet ; come out , dear sun , the world to greet !
[S34] and no one thought of the snow - man .

## Sample 89

**story_name:** farquhar-macneill  
**question:** why did farquhar strike against the chimney and stuck fast in the thatch ?  
**answer:** he did not look where he was going .  

**old sentence count:** 7  
**new sentence count:** 7  

### Old Split

[S0] all would have gone well if farquhar had only looked where he was going .
[S1] he did not , being deeply engaged in making love to a young fairy maiden by his side , so he never saw a cottage that was standing right in his way .
[S2] he struck against the chimney and stuck fast in the thatch .
[S3] his companions sped merrily on , not noticing what had befallen him .
[S4] he was left to disentangle himself as best he could .
[S5] as he was doing so he chanced to glance down the wide chimney .
[S6] in the cottage kitchen he saw a comely young woman dandling a rosy - cheeked baby .

### New Split

[S0] all would have gone well if farquhar had only looked where he was going .
[S1] he did not , being deeply engaged in making love to a young fairy maiden by his side , so he never saw a cottage that was standing right in his way .
[S2] he struck against the chimney and stuck fast in the thatch .
[S3] his companions sped merrily on , not noticing what had befallen him .
[S4] he was left to disentangle himself as best he could .
[S5] as he was doing so he chanced to glance down the wide chimney .
[S6] in the cottage kitchen he saw a comely young woman dandling a rosy - cheeked baby .

## Sample 90

**story_name:** the-fire-plume  
**question:** why was the cousin worried about wassamo's disappearance ?  
**answer:** he cared for wassamo .  

**old sentence count:** 33  
**new sentence count:** 31  

### Old Split

[S0] as he opened his eyes , in a dreamy way , he saw the kettle near him .
[S1] some of the fish he observed were in the bowl .
[S2] the fire flickered , and made light and shadow ; but nowhere was wassamo to be seen .
[S3] he waited , and waited again , in the expectation that wassamo would appear .
[S4] " perhaps , " thought the cousin , " he is gone out again to visit the nets .
[S5] " he looked off that way , but the canoe still lay close by the rock at the shore .
[S6] he searched and found his footsteps in the ashes , and out upon the green ground a little distance , and then they were utterly lost .
[S7] he was now greatly troubled in spirit , and he called aloud , " netawis !
[S8] cousin !
[S9] cousin !
[S10] " but there was no answer to his call .
[S11] he called again in his sorrow , louder and louder , " netawis !
[S12] netawis !
[S13] cousin !
[S14] cousin !
[S15] whither are you gone ?
[S16] " but no answer came to his voice of wailing .
[S17] he started for the edge of the woods , crying as he ran , " my cousin !
[S18] " and " oh , my cousin !
[S19] " hither and thither through the forest he sped with all his fleetness of foot and quickness of spirit ; and when at last he found that no voice would answer him , he burst into tears , and sobbed aloud .
[S20] he returned to the fire , and sat down .
[S21] he mused upon the absence of wassamo with a sorely - troubled heart .
[S22] " he may have been playing me a trick , " he thought ; but it was full time that the trick should be at an end , and wassamo returned not .
[S23] the cousin cherished other hopes , but they all died away in the morning light , when he found himself alone by the hunting - fire .
[S24] " how shall i answer to his friends for wassamo ?
[S25] " thought the cousin .
[S26] " although , " he said to himself , " his parents are my kindred , and they are well assured that their son is my bosom - friend , will they receive that belief in the place of him who is lost .
[S27] no , no ; they will say that i have slain him , and they will require blood for blood .
[S28] oh !
[S29] my cousin , whither are you gone ?
[S30] " he would have rested to restore his mind to its peace , but he could not sleep ; and , without further regard to net or canoe , he set off for the village , running all the way .
[S31] as they saw him approaching at such speed and alone , they said , " some accident has happened .
[S32] "

### New Split

[S0] as he opened his eyes , in a dreamy way , he saw the kettle near him .
[S1] some of the fish he observed were in the bowl .
[S2] the fire flickered , and made light and shadow ; but nowhere was wassamo to be seen .
[S3] he waited , and waited again , in the expectation that wassamo would appear .
[S4] " perhaps , " thought the cousin , " he is gone out again to visit the nets . "
[S5] he looked off that way , but the canoe still lay close by the rock at the shore .
[S6] he searched and found his footsteps in the ashes , and out upon the green ground a little distance , and then they were utterly lost .
[S7] he was now greatly troubled in spirit , and he called aloud , " netawis ! "
[S8] " cousin ! "
[S9] " cousin ! "
[S10] but there was no answer to his call .
[S11] he called again in his sorrow , louder and louder , " netawis ! "
[S12] " netawis ! "
[S13] " cousin ! "
[S14] " cousin ! "
[S15] " whither are you gone ? "
[S16] but no answer came to his voice of wailing .
[S17] he started for the edge of the woods , crying as he ran , " my cousin ! "
[S18] and " oh , my cousin ! "
[S19] hither and thither through the forest he sped with all his fleetness of foot and quickness of spirit ; and when at last he found that no voice would answer him , he burst into tears , and sobbed aloud .
[S20] he returned to the fire , and sat down .
[S21] he mused upon the absence of wassamo with a sorely - troubled heart .
[S22] " he may have been playing me a trick , " he thought ; but it was full time that the trick should be at an end , and wassamo returned not .
[S23] the cousin cherished other hopes , but they all died away in the morning light , when he found himself alone by the hunting - fire .
[S24] " how shall i answer to his friends for wassamo ? " thought the cousin .
[S25] " although , " he said to himself , " his parents are my kindred , and they are well assured that their son is my bosom - friend , will they receive that belief in the place of him who is lost . "
[S26] " no , no ; they will say that i have slain him , and they will require blood for blood . "
[S27] " oh ! "
[S28] " my cousin , whither are you gone ? "
[S29] he would have rested to restore his mind to its peace , but he could not sleep ; and , without further regard to net or canoe , he set off for the village , running all the way .
[S30] as they saw him approaching at such speed and alone , they said , " some accident has happened . "

## Sample 91

**story_name:** ogre-of-rashomon  
**question:** how did people feel about raiko ?  
**answer:** respect .  

**old sentence count:** 3  
**new sentence count:** 3  

### Old Split

[S0] now at this time there lived in kyoto a general named raiko , who had made himself famous for his brave deeds .
[S1] some time before this he made the country ring with his name , for he had attacked oeyama , where a band of ogres lived with their chief , who instead of wine drank the blood of human beings .
[S2] he had routed them all and cut off the head of the chief monster

### New Split

[S0] now at this time there lived in kyoto a general named raiko , who had made himself famous for his brave deeds .
[S1] some time before this he made the country ring with his name , for he had attacked oeyama , where a band of ogres lived with their chief , who instead of wine drank the blood of human beings .
[S2] he had routed them all and cut off the head of the chief monster

## Sample 92

**story_name:** flax  
**question:** why is the paper so happy ?  
**answer:** because splendid thoughts are written on it .  

**old sentence count:** 9  
**new sentence count:** 8  

### Old Split

[S0] " i never imagined anything like this when i was only a little blue flower growing in the fields , " said the paper .
[S1] " how could i know that i should ever be the means of bringing knowledge and joy to men ?
[S2] i can not understand it myself , and yet it is really so .
[S3] heaven knows that i have done nothing myself but what i was obliged to do with my weak powers for my own preservation ; and yet i have been promoted from one joy and honor to another .
[S4] each time i think that the song is ended , and then something higher and better begins for me .
[S5] i suppose now i shall be sent out to journey about the world , so that people may read me .
[S6] it can not be otherwise , for i have more splendid thoughts written upon me than i had pretty flowers in olden times .
[S7] i am happier than ever .
[S8] "

### New Split

[S0] " i never imagined anything like this when i was only a little blue flower growing in the fields , " said the paper .
[S1] " how could i know that i should ever be the means of bringing knowledge and joy to men ? "
[S2] " i can not understand it myself , and yet it is really so . "
[S3] " heaven knows that i have done nothing myself but what i was obliged to do with my weak powers for my own preservation ; and yet i have been promoted from one joy and honor to another . "
[S4] " each time i think that the song is ended , and then something higher and better begins for me . "
[S5] " i suppose now i shall be sent out to journey about the world , so that people may read me . "
[S6] " it can not be otherwise , for i have more splendid thoughts written upon me than i had pretty flowers in olden times . "
[S7] " i am happier than ever . "

## Sample 93

**story_name:** the-fire-plume  
**question:** what made the village not doubt the cousin's word in anything ?  
**answer:** the cousin told the truth in the past .  

**old sentence count:** 9  
**new sentence count:** 9  

### Old Split

[S0] they then took an affectionate leave of each other , wassamo enjoining it upon his cousin , at risk of his life , to not look back when he had once started to return .
[S1] the cousin , sore at heart , but constrained to obey , parted from them , and as he walked sadly away , he heard a gliding noise as of the sound of waters that were cleaved .
[S2] he returned home , and told his friends that wassamo and his wife had disappeared , but that he knew not how .
[S3] no one doubted his word in any thing now .
[S4] wassamo with his wife soon reached their home at the hills .
[S5] the old sand - spirit was in excellent health , and delighted to see them .
[S6] he hailed their return with open arms ; and he opened his arms so very wide , that when he closed them he not only embraced wassamo and his wife , but all of the tobacco - sacks which they had brought with them .
[S7] the requests of the indian people were made known to him ; he replied that he would attend to all , but that he must first invite his friends to smoke with him .
[S8] accordingly he at once dispatched his pipe - bearer and confidential aid to summon various spirits of his acquaintance , and set the time for them to come .

### New Split

[S0] they then took an affectionate leave of each other , wassamo enjoining it upon his cousin , at risk of his life , to not look back when he had once started to return .
[S1] the cousin , sore at heart , but constrained to obey , parted from them , and as he walked sadly away , he heard a gliding noise as of the sound of waters that were cleaved .
[S2] he returned home , and told his friends that wassamo and his wife had disappeared , but that he knew not how .
[S3] no one doubted his word in any thing now .
[S4] wassamo with his wife soon reached their home at the hills .
[S5] the old sand - spirit was in excellent health , and delighted to see them .
[S6] he hailed their return with open arms ; and he opened his arms so very wide , that when he closed them he not only embraced wassamo and his wife , but all of the tobacco - sacks which they had brought with them .
[S7] the requests of the indian people were made known to him ; he replied that he would attend to all , but that he must first invite his friends to smoke with him .
[S8] accordingly he at once dispatched his pipe - bearer and confidential aid to summon various spirits of his acquaintance , and set the time for them to come .

## Sample 94

**story_name:** bokwewa-the-humpback  
**question:** why did kwasynd ask bokwewa to restore the beautiful young woman ?  
**answer:** she was dead .  

**old sentence count:** 11  
**new sentence count:** 10  

### Old Split

[S0] he traveled for a long time .
[S1] at length he fell in with the footsteps of men .
[S2] they were moving by encampments , for he saw , at several spots , the poles where they had passed .
[S3] it was winter ; and coming to a place where one of their company had died , he found upon a scaffold , lying at length in the cold blue air , the body of a beautiful young woman .
[S4] " she shall be my wife !
[S5] " exclaimed kwasynd .
[S6] he lifted her up , and bearing her in his arms , he returned to his brother .
[S7] " brother , " he said , " can not you restore her to life ?
[S8] oh , do me that favor !
[S9] " he looked upon the beautiful female with a longing gaze ; but she lay as cold and silent as when he had found her upon the scaffold .
[S10] " i will try , " said bokwewa .

### New Split

[S0] he traveled for a long time .
[S1] at length he fell in with the footsteps of men .
[S2] they were moving by encampments , for he saw , at several spots , the poles where they had passed .
[S3] it was winter ; and coming to a place where one of their company had died , he found upon a scaffold , lying at length in the cold blue air , the body of a beautiful young woman .
[S4] " she shall be my wife ! " exclaimed kwasynd .
[S5] he lifted her up , and bearing her in his arms , he returned to his brother .
[S6] " brother , " he said , " can not you restore her to life ? "
[S7] " oh , do me that favor ! "
[S8] he looked upon the beautiful female with a longing gaze ; but she lay as cold and silent as when he had found her upon the scaffold .
[S9] " i will try , " said bokwewa .

## Sample 95

**story_name:** secret-church  
**question:** how did the pastor and his family treat the schoolmaster ?  
**answer:** kindly .  

**old sentence count:** 10  
**new sentence count:** 10  

### Old Split

[S0] after a time the priest came by , and he was so old and decrepit that his wife and daughter led him .
[S1] and when they came to the spot where the schoolmaster was standing , they stopped and invited him to come to church and hear mass .
[S2] the schoolmaster thought for a moment ; but since it occurred to him that it might be amusing to see how these people worshiped god , he said he would go along , if he did not thereby suffer harm .
[S3] no , no harm should come to him , said they , but rather a blessing .
[S4] in the church all went forward in a quiet and orderly manner , there were neither dogs nor crying children to disturb the service , and the singing was good -- but he could not make out the words .
[S5] when the priest had been led to the pulpit he delivered what seemed to the listening schoolmaster a really fine and edifying sermon -- but one , it appeared to him , of quite a peculiar trend of thought , which he was not always able to follow .
[S6] nor did the " our father in heaven ...
[S7] " sound just right , and the " deliver us from evil ...
[S8] " he did not hear at all .
[S9] nor was the name of jesus uttered ; and at the close no blessing was spoken .

### New Split

[S0] after a time the priest came by , and he was so old and decrepit that his wife and daughter led him .
[S1] and when they came to the spot where the schoolmaster was standing , they stopped and invited him to come to church and hear mass .
[S2] the schoolmaster thought for a moment ; but since it occurred to him that it might be amusing to see how these people worshiped god , he said he would go along , if he did not thereby suffer harm .
[S3] no , no harm should come to him , said they , but rather a blessing .
[S4] in the church all went forward in a quiet and orderly manner , there were neither dogs nor crying children to disturb the service , and the singing was good -- but he could not make out the words .
[S5] when the priest had been led to the pulpit he delivered what seemed to the listening schoolmaster a really fine and edifying sermon -- but one , it appeared to him , of quite a peculiar trend of thought , which he was not always able to follow .
[S6] nor did the " our father in heaven ... "
[S7] sound just right , and the " deliver us from evil ... "
[S8] he did not hear at all .
[S9] nor was the name of jesus uttered ; and at the close no blessing was spoken .

## Sample 96

**story_name:** brave-tin-soldier  
**question:** what did the cook do when she found the soldier ?  
**answer:** brough him back to the play room .  

**old sentence count:** 6  
**new sentence count:** 6  

### Old Split

[S0] the fish swam to and fro , making the most wonderful movements , but at last he became quite still .
[S1] after a while , a flash of lightning seemed to pass through him , and then the daylight approached , and a voice cried out , " i declare here is the tin soldier .
[S2] " the fish had been caught , taken to the market and sold to the cook , who took him into the kitchen and cut him open with a large knife .
[S3] she picked up the soldier and held him by the waist between her finger and thumb , and carried him into the room .
[S4] they were all anxious to see this wonderful soldier who had travelled about inside a fish ; but he was not at all proud .
[S5] they placed him on the table , and -- how many curious things do happen in the world!--there he was in the very same room from the window of which he had fallen , there were the same children , the same playthings , standing on the table , and the pretty castle with the elegant little dancer at the door ; she still balanced herself on one leg , and held up the other , so she was as firm as himself .

### New Split

[S0] the fish swam to and fro , making the most wonderful movements , but at last he became quite still .
[S1] after a while , a flash of lightning seemed to pass through him , and then the daylight approached , and a voice cried out , " i declare here is the tin soldier . "
[S2] the fish had been caught , taken to the market and sold to the cook , who took him into the kitchen and cut him open with a large knife .
[S3] she picked up the soldier and held him by the waist between her finger and thumb , and carried him into the room .
[S4] they were all anxious to see this wonderful soldier who had travelled about inside a fish ; but he was not at all proud .
[S5] they placed him on the table , and -- how many curious things do happen in the world!--there he was in the very same room from the window of which he had fallen , there were the same children , the same playthings , standing on the table , and the pretty castle with the elegant little dancer at the door ; she still balanced herself on one leg , and held up the other , so she was as firm as himself .

## Sample 97

**story_name:** soria-moria-castle  
**question:** why could halvor hardly keep up with the west wind ?  
**answer:** the west wind was too quick .  

**old sentence count:** 15  
**new sentence count:** 14  

### Old Split

[S0] halvor was very restless , and wanted to go right on again , but the woman said there was no need to hurry .
[S1] " lie down on the bench by the stove , and take a nap , for we have no bed for you , " said she .
[S2] " i will watch for the west wind 's coming .
[S3] " all of a sudden the west wind came rushing along so that the walls creaked .
[S4] the woman ran out : " you west wind !
[S5] you west wind !
[S6] can you tell me the way to soria - moria castle ?
[S7] there is a fellow here who wants to know .
[S8] " " yes , indeed , " said the west wind , " i have to go to that very place , and dry the wash for the wedding soon to be held .
[S9] if he is quick afoot , he may come along with me .
[S10] " halvor ran out .
[S11] " you must hurry if you are going with me , " said the west wind ; and at once he was up and off over hill and dale , land and sea , so that halvor could hardly keep up with him .
[S12] " now i have no more time to keep you company , " said the west wind , " because i have first to tear down a stretch of pine forest , before i come to the bleaching - field and dry the wash .
[S13] but if you keep going along the hills , you will meet some girls standing there and washing , and then you will not be far from soria - moria castle .
[S14] "

### New Split

[S0] halvor was very restless , and wanted to go right on again , but the woman said there was no need to hurry .
[S1] " lie down on the bench by the stove , and take a nap , for we have no bed for you , " said she .
[S2] " i will watch for the west wind 's coming . "
[S3] all of a sudden the west wind came rushing along so that the walls creaked .
[S4] the woman ran out : " you west wind ! "
[S5] " you west wind ! "
[S6] " can you tell me the way to soria - moria castle ? "
[S7] " there is a fellow here who wants to know . "
[S8] " yes , indeed , " said the west wind , " i have to go to that very place , and dry the wash for the wedding soon to be held . "
[S9] " if he is quick afoot , he may come along with me . "
[S10] halvor ran out .
[S11] " you must hurry if you are going with me , " said the west wind ; and at once he was up and off over hill and dale , land and sea , so that halvor could hardly keep up with him .
[S12] " now i have no more time to keep you company , " said the west wind , " because i have first to tear down a stretch of pine forest , before i come to the bleaching - field and dry the wash . "
[S13] " but if you keep going along the hills , you will meet some girls standing there and washing , and then you will not be far from soria - moria castle . "

## Sample 98

**story_name:** the-black-bull-of-norroway  
**question:** why did the youngest princess want to find out her fortune ?  
**answer:** her sisters ' fortunes were good .  

**old sentence count:** 9  
**new sentence count:** 9  

### Old Split

[S0] and i 've heard tell that they drew her to the palace of a great and wealthy prince , who married her ; but that is outside my story .
[S1] a few weeks afterwards , the second princess thought that she would do as her sister had done , and go down to the hen - wife 's cottage , and tell her that she , too , was going out into the world to seek her fortune .
[S2] and , of course , in her heart of hearts she hoped that what had happened to her sister would happen to her also .
[S3] and , curious to say , it did .
[S4] for the old hen - wife sent her to look out at her back door , and she went , and , lo and behold !
[S5] another coach - and - six was coming along the road .
[S6] and when she went and told the old woman , she smiled upon her kindly , and told her to hurry home , for the coach - and - six was her fortune also , and that it had come for her .
[S7] so she , too , ran home , and got into her grand carriage , and was driven away .
[S8] and , of course , after all these lucky happenings , the youngest princess was anxious to try what her fortune might be ; so the very night , in high good humour , she tripped away down to the old witch 's cottage .

### New Split

[S0] and i 've heard tell that they drew her to the palace of a great and wealthy prince , who married her ; but that is outside my story .
[S1] a few weeks afterwards , the second princess thought that she would do as her sister had done , and go down to the hen - wife 's cottage , and tell her that she , too , was going out into the world to seek her fortune .
[S2] and , of course , in her heart of hearts she hoped that what had happened to her sister would happen to her also .
[S3] and , curious to say , it did .
[S4] for the old hen - wife sent her to look out at her back door , and she went , and , lo and behold !
[S5] another coach - and - six was coming along the road .
[S6] and when she went and told the old woman , she smiled upon her kindly , and told her to hurry home , for the coach - and - six was her fortune also , and that it had come for her .
[S7] so she , too , ran home , and got into her grand carriage , and was driven away .
[S8] and , of course , after all these lucky happenings , the youngest princess was anxious to try what her fortune might be ; so the very night , in high good humour , she tripped away down to the old witch 's cottage .

## Sample 99

**story_name:** the-crane-that-crossed-the-river  
**question:** how will the sons and their father feel seeing the skull ?  
**answer:** terrified .  

**old sentence count:** 20  
**new sentence count:** 19  

### Old Split

[S0] changed , but the same , with ghastly looks and arms that were withered , she appeared to her sons as they returned from the hunt , in the twilight , in the close of the day .
[S1] at night she darkly unlatched the lodge - door and glided in , and bent over them as they sought to sleep .
[S2] oftenest it was her bare brow , white , and bony , and bodyless , that they saw floating in the air , and making a mock of them in the wild paths of the forest , or in the midnight darkness of the lodge .
[S3] she was a terror to all their lives , and she made every spot where they had seen her , hideous to the living eye ; so that after being long buffeted and beset , they at last resolved , together with their father , now stricken in years , to leave the country .
[S4] they began a journey toward the south .
[S5] after traveling many days along the shore of a great lake , they passed around a craggy bluff , and came upon a scene where there was a rough fall of waters , and a river issuing forth from the lake .
[S6] they had no sooner come in sight of this fall of water , than they heard a rolling sound behind them , and looking back , they beheld the skull of a woman rolling along the beach .
[S7] it seemed to be pursuing them , and it came on with great speed ; when , behold , from out of the woods hard by , appeared a headless body , which made for the beach with the utmost dispatch .
[S8] the skull too advanced toward it , and when they looked again , lo !
[S9] they had united , and were making all haste to come up with the hunter and his two sons .
[S10] they now might well be in extreme fear , for they knew not how to escape her .
[S11] at this moment , one of them looked out and saw a stately crane sitting on a rock in the middle of the rapids .
[S12] they called out to the bird , " see , grandfather , we are persecuted .
[S13] come and take us across the falls that we may escape her .
[S14] " the crane so addressed was of extraordinary size , and had arrived at a great old age , and , as might be expected , he sat , when first descried by the two sons , in a state of profound thought , revolving his long experience of life there in the midst of the most violent eddies .
[S15] when he heard himself appealed to , the crane stretched forth his neck with great deliberation , and lifting himself slowly by his wings , he flew across to their assistance .
[S16] " be careful , " said the old crane , " that you do not touch the crown of my head .
[S17] i am bald from age and long service , and very tender at that spot .
[S18] should you be so unlucky as to lay a hand upon it , i shall not be able to avoid throwing you both in the rapids .
[S19] "

### New Split

[S0] changed , but the same , with ghastly looks and arms that were withered , she appeared to her sons as they returned from the hunt , in the twilight , in the close of the day .
[S1] at night she darkly unlatched the lodge - door and glided in , and bent over them as they sought to sleep .
[S2] oftenest it was her bare brow , white , and bony , and bodyless , that they saw floating in the air , and making a mock of them in the wild paths of the forest , or in the midnight darkness of the lodge .
[S3] she was a terror to all their lives , and she made every spot where they had seen her , hideous to the living eye ; so that after being long buffeted and beset , they at last resolved , together with their father , now stricken in years , to leave the country .
[S4] they began a journey toward the south .
[S5] after traveling many days along the shore of a great lake , they passed around a craggy bluff , and came upon a scene where there was a rough fall of waters , and a river issuing forth from the lake .
[S6] they had no sooner come in sight of this fall of water , than they heard a rolling sound behind them , and looking back , they beheld the skull of a woman rolling along the beach .
[S7] it seemed to be pursuing them , and it came on with great speed ; when , behold , from out of the woods hard by , appeared a headless body , which made for the beach with the utmost dispatch .
[S8] the skull too advanced toward it , and when they looked again , lo !
[S9] they had united , and were making all haste to come up with the hunter and his two sons .
[S10] they now might well be in extreme fear , for they knew not how to escape her .
[S11] at this moment , one of them looked out and saw a stately crane sitting on a rock in the middle of the rapids .
[S12] they called out to the bird , " see , grandfather , we are persecuted . "
[S13] " come and take us across the falls that we may escape her . "
[S14] the crane so addressed was of extraordinary size , and had arrived at a great old age , and , as might be expected , he sat , when first descried by the two sons , in a state of profound thought , revolving his long experience of life there in the midst of the most violent eddies .
[S15] when he heard himself appealed to , the crane stretched forth his neck with great deliberation , and lifting himself slowly by his wings , he flew across to their assistance .
[S16] " be careful , " said the old crane , " that you do not touch the crown of my head . "
[S17] " i am bald from age and long service , and very tender at that spot . "
[S18] " should you be so unlucky as to lay a hand upon it , i shall not be able to avoid throwing you both in the rapids . "

## Sample 100

**story_name:** thomas-the-rhymer  
**question:** what will happen after the queen leaves for fairy-land ?  
**answer:** thomas will become famous for his powers .  

**old sentence count:** 8  
**new sentence count:** 8  

### Old Split

[S0] once more the grey palfrey was brought , and thomas and the queen mounted it ; and , as they had come , so they returned to the eildon tree near the huntly burn .
[S1] then the queen bade thomas farewell ; and , as a parting gift , he asked her to give him something that would let people know that he had really been to fairy - land .
[S2] " i have already given thee the gift of truth , " she replied .
[S3] " i will now give thee the gifts of prophecy and poesie ; so that thou wilt be able to foretell the future , and also to write wondrous verses .
[S4] and , besides these unseen gifts , here is something that mortals can see with their own eyes -- a harp that was fashioned in fairy - land .
[S5] fare thee well , my friend .
[S6] some day , perchance , i will return for thee again .
[S7] " with these words the lady vanished , and thomas was left alone , feeling a little sorry , if the truth must be told , at parting with such a radiant being and coming back to the ordinary haunts of men .

### New Split

[S0] once more the grey palfrey was brought , and thomas and the queen mounted it ; and , as they had come , so they returned to the eildon tree near the huntly burn .
[S1] then the queen bade thomas farewell ; and , as a parting gift , he asked her to give him something that would let people know that he had really been to fairy - land .
[S2] " i have already given thee the gift of truth , " she replied .
[S3] " i will now give thee the gifts of prophecy and poesie ; so that thou wilt be able to foretell the future , and also to write wondrous verses . "
[S4] " and , besides these unseen gifts , here is something that mortals can see with their own eyes -- a harp that was fashioned in fairy - land . "
[S5] " fare thee well , my friend . "
[S6] " some day , perchance , i will return for thee again . "
[S7] with these words the lady vanished , and thomas was left alone , feeling a little sorry , if the truth must be told , at parting with such a radiant being and coming back to the ordinary haunts of men .
