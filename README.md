### DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python
***reinforcement learning Double Deep Q Learning (DDQN) method to solve OpenAi Gym "LunarLander-v2" by usnig Double Deep NeuralNetworks
Solving Openai gym LunarLanderV2 by using Double DQN***  
Naive TensorFlow (using Keras) implementation of paper:  Deep Reinforcement Learning with Double Q-learning
https://arxiv.org/abs/1509.06461   

[feel free to ask any question in Issues or just email me]  
Mohammad.E.Asadolahi@gmail.com

#### How to install requirements
The `requirements.txt` file should list all Python libraries that the project depend on, and they will be installed using:
```
pip install -r Requirements.txt
```
I keep updating the project to be compatible with new versions of libraries. If there was any problem with the diffrent versions of the required libraries let me know in the "Issues" section, so i can resolve them.  
  
  
**this code is implemented with TensorFlow using Keras! i will add Pytorch version soon!!**  
#### to do:  
* deploy the ReplayBuffer code   [***done***]
* add imports file   [***done***]
* deploy Q approximation neural network [***done***]
* deploy the Agnet in pytorch  [***done***]
* deploy the Environmetn class (using OpenAI Gym library to import an environment) [***done***]
* deploy the main learning loop [***done***]
* deploy the test loop [***done***]
* run a standard RL evaluation
* refactor the project in Pytorch

#### Sample Training process
episode: 2   reward: -123.4838284048032  avg so far:-123.4838284048032 exploreRate:0.9860940917766235  
episode: 3   reward: -120.2480775877887  avg so far:-121.86595299629596 exploreRate:0.9333105749632893  
episode: 4   reward: -124.38843377713812  avg so far:-122.70677992324335 exploreRate:0.8985016297353923  
episode: 5   reward: -172.6867510698475  avg so far:-135.20177270989439 exploreRate:0.8542431659050345  
episode: 6   reward: -185.0293336390434  avg so far:-145.1672848957242 exploreRate:0.821971943634938  
episode: 7   reward: -37.67185613154821  avg so far:-127.25138010169486 exploreRate:0.7948853455740398  
episode: 8   reward: -268.06423176214395  avg so far:-147.36750176747333 exploreRate:0.7561088752427627  
episode: 9   reward: -425.7797816503208  avg so far:-182.16903675282924 exploreRate:0.717786647345383  
episode: 10   reward: -11.2537898917019  avg so far:-163.17845376825954 exploreRate:0.6834545255907595  
episode: 11   reward: -238.72555770217554  avg so far:-170.73316416165113 exploreRate:0.6484902655231009  
episode: 12   reward: -301.5535697860296  avg so far:-182.62592830932192 exploreRate:0.6025246707069989  
episode: 13   reward: -120.77281332623825  avg so far:-177.4715020607316 exploreRate:0.582378225659823  
episode: 14   reward: -119.89580741154116  avg so far:-173.04260247233236 exploreRate:0.5470851615600282  
episode: 15   reward: -372.0865336932594  avg so far:-187.26002613096998 exploreRate:0.510089893972876  
episode: 16   reward: -93.45911570357939  avg so far:-181.00663210247728 exploreRate:0.48496384592188013  
episode: 17   reward: -35.09151153467246  avg so far:-171.88693706698947 exploreRate:0.45762946982477415  
episode: 18   reward: -80.14800461689539  avg so far:-166.49052927580746 exploreRate:0.42668341995630865  
episode: 19   reward: -314.54025339957724  avg so far:-174.71551394935022 exploreRate:0.39406759337370495  
episode: 20   reward: -118.19994869634078  avg so far:-171.74101051498133 exploreRate:0.37353401519915735  
episode: 21   reward: -116.24634455781487  avg so far:-168.966277217123 exploreRate:0.35054643911827926  
episode: 22   reward: -82.73810805575839  avg so far:-164.8601739237247 exploreRate:0.3274961128836704  
episode: 23   reward: -64.89910345705493  avg so far:-160.31648890251242 exploreRate:0.30626766148829976  
episode: 24   reward: -92.52296798330724  avg so far:-157.3689445147209 exploreRate:0.28299795565300107  
episode: 25   reward: -64.2411211811071  avg so far:-153.488618542487 exploreRate:0.26544924862125363  
episode: 26   reward: -49.69368793715303  avg so far:-149.33682131827362 exploreRate:0.24824270402236964  
episode: 27   reward: -81.74180480107557  avg so far:-146.7370129906891 exploreRate:0.23331545165791032  
episode: 28   reward: 19.978629408119318  avg so far:-140.562359568511 exploreRate:0.2093200192862765  
episode: 29   reward: -105.79809762224113  avg so far:-139.32077878471566 exploreRate:0.19565392171799403  
episode: 30   reward: -52.19845779691747  avg so far:-136.31656081961916 exploreRate:0.16772217330870298  
episode: 31   reward: -74.80626993435051  avg so far:-134.2662177901102 exploreRate:0.14464346723516094  
episode: 32   reward: -49.99014180543845  avg so far:-131.54763469383047 exploreRate:0.12612047569284585  
episode: 33   reward: -74.09642389466256  avg so far:-129.75228435635645 exploreRate:0.10581515136579583  
episode: 34   reward: -63.25459392612083  avg so far:-127.7372028281675 exploreRate:0.0968023682492183  
episode: 35   reward: -111.5082942302834  avg so far:-127.25988198705326 exploreRate:0.08715125746578377  
episode: 36   reward: -21.55929648606306  avg so far:-124.23986525845355 exploreRate:0.08142059044946617  
episode: 37   reward: 17.415706373455663  avg so far:-120.3049882686783 exploreRate:0.0663256650993032  
episode: 38   reward: -13.633420152562465  avg so far:-117.42197291418867 exploreRate:0.05882382252580734  
episode: 39   reward: -53.95772668203037  avg so far:-115.75186117123715 exploreRate:0.05277406008946195  
episode: 40   reward: -18.807254372609236  avg so far:-113.26610202255438 exploreRate:0.04775075008886719  
episode: 41   reward: -90.1435829441037  avg so far:-112.68803904559311 exploreRate:0.044056576776103695  
episode: 42   reward: -344.58048658440794  avg so far:-118.34395240019836 exploreRate:0.0371488198889033  
episode: 43   reward: -143.4658351600295  avg so far:-118.94209246590862 exploreRate:0.035090054747437864  
episode: 44   reward: -34.86553444391481  avg so far:-116.98682367469947 exploreRate:0.03250515260355337  
episode: 45   reward: -33.83159474433364  avg so far:-115.09693210810023 exploreRate:0.03068838482755771  
episode: 46   reward: -15.02435242730786  avg so far:-112.87309700408264 exploreRate:0.028130727419394437  
episode: 47   reward: -63.5600690831217  avg so far:-111.80107465797478 exploreRate:0.026518635876378482  
episode: 48   reward: 15.586223248135681  avg so far:-109.09070661741924 exploreRate:0.02477489199529718  
episode: 49   reward: -43.31236160001143  avg so far:-107.72032442955657 exploreRate:0.023507454683176096  
episode: 50   reward: 5.090893789615464  avg so far:-105.41805466998163 exploreRate:0.022071823407932747  
episode: 51   reward: 2.6476389620282106  avg so far:-103.25674079734142 exploreRate:0.020703149341900956  
episode: 52   reward: 15.230328233738462  avg so far:-100.93346493398691 exploreRate:0.018920827951510052  
episode: 53   reward: -30.042970973491222  avg so far:-99.57018620397737 exploreRate:0.01747451339040307  
episode: 54   reward: -40.21244703906416  avg so far:-98.45022886124316 exploreRate:0.010597502922393745  
episode: 55   reward: -60.26020356540108  avg so far:-97.74300617057942 exploreRate:0.01  
episode: 56   reward: -229.1775756189617  avg so far:-100.13272561509547 exploreRate:0.01  
episode: 57   reward: -124.18460070220631  avg so far:-100.56222338450814 exploreRate:0.01  
episode: 58   reward: -132.32633617418344  avg so far:-101.11948852116912 exploreRate:0.01  
episode: 59   reward: -2.6926712781100406  avg so far:-99.42247443077154 exploreRate:0.01  
episode: 60   reward: -127.66049807723057  avg so far:-99.90108500105052 exploreRate:0.01  
episode: 61   reward: -52.15468270479044  avg so far:-99.10531162944618 exploreRate:0.01  
episode: 62   reward: -247.4750980113139  avg so far:-101.5376032094768 exploreRate:0.01  
episode: 63   reward: -88.40214517307751  avg so far:-101.32574098308326 exploreRate:0.01  
episode: 64   reward: -144.72454798438207  avg so far:-102.01461093548484 exploreRate:0.01  
episode: 65   reward: -162.36282009341164  avg so far:-102.95755170357745 exploreRate:0.01  
episode: 66   reward: -241.54922095640123  avg so far:-105.08973123054396 exploreRate:0.01  
episode: 67   reward: -132.91530760243324  avg so far:-105.51133087254229 exploreRate:0.01  
episode: 68   reward: -117.53966627859563  avg so far:-105.69085826666247 exploreRate:0.01  
episode: 69   reward: -113.82848094858642  avg so far:-105.81052918845548 exploreRate:0.01  
episode: 70   reward: -75.45749534832099  avg so far:-105.37063014729411 exploreRate:0.01  
episode: 71   reward: -201.9707178576109  avg so far:-106.75063140029863 exploreRate:0.01  
episode: 72   reward: -125.84534751789522  avg so far:-107.01957106392675 exploreRate:0.01  
episode: 73   reward: 146.21845925656942  avg so far:-103.5023761983643 exploreRate:0.01  
episode: 74   reward: -72.3793011776562  avg so far:-103.07603270492994 exploreRate:0.01  
episode: 75   reward: -51.74926894528299  avg so far:-102.38242778925904 exploreRate:0.01  
episode: 76   reward: 135.06297173539136  avg so far:-99.21648912893038 exploreRate:0.01  
episode: 77   reward: 93.9429292345178  avg so far:-96.67491783467447 exploreRate:0.01  
episode: 78   reward: -26.27937796599307  avg so far:-95.76069004417212 exploreRate:0.01  
episode: 79   reward: 223.86527249263636  avg so far:-91.66292129370022 exploreRate:0.01  
episode: 80   reward: 2.7618457448884612  avg so far:-90.46767107802188 exploreRate:0.01  
episode: 81   reward: -114.37936354501174  avg so far:-90.76656723385926 exploreRate:0.01  
episode: 82   reward: -4.965413173904286  avg so far:-89.70729372694623 exploreRate:0.01  
episode: 83   reward: 158.71869069253864  avg so far:-86.67770855109886 exploreRate:0.01  
episode: 84   reward: 179.82488709775214  avg so far:-83.46683390472715 exploreRate:0.01  
episode: 85   reward: -98.27096885519394  avg so far:-83.64307360651843 exploreRate:0.01  
episode: 86   reward: -48.58030069210426  avg so far:-83.23057039576061 exploreRate:0.01  
episode: 87   reward: -93.11430788989612  avg so far:-83.34549757592498 exploreRate:0.01  
episode: 88   reward: -67.88553032133538  avg so far:-83.16779680288371 exploreRate:0.01  
episode: 89   reward: 226.97088341654666  avg so far:-79.643493618572 exploreRate:0.01  
episode: 90   reward: -38.37155376780055  avg so far:-79.17976395732738 exploreRate:0.01  
episode: 91   reward: 222.41985785632707  avg so far:-75.82865704828677 exploreRate:0.01  
episode: 92   reward: -114.47854830394733  avg so far:-76.25338112801931 exploreRate:0.01  
episode: 93   reward: -102.34688684383043  avg so far:-76.5370061901477 exploreRate:0.01  
episode: 94   reward: -78.43013325382164  avg so far:-76.55736239513344 exploreRate:0.01  
episode: 95   reward: -57.36324728630183  avg so far:-76.35316968120969 exploreRate:0.01  
episode: 96   reward: -108.2988989947929  avg so far:-76.68944051608952 exploreRate:0.01  
episode: 97   reward: 24.63757233901933  avg so far:-75.6339507988488 exploreRate:0.01  
episode: 98   reward: -56.51819178492001  avg so far:-75.43688111829283 exploreRate:0.01  
episode: 99   reward: -95.59611355176675  avg so far:-75.64258757169563 exploreRate:0.01  
episode: 100   reward: -48.1395787626832  avg so far:-75.36477940190763 exploreRate:0.01  
episode: 101   reward: -158.56379114280878  avg so far:-76.19676951931663 exploreRate:0.01  
episode: 102   reward: 3.8384166714230616  avg so far:-75.4043419332697 exploreRate:0.01  
episode: 103   reward: -73.48217446295344  avg so far:-75.38549715414895 exploreRate:0.01  
episode: 104   reward: 205.57720318751095  avg so far:-72.65770394694837 exploreRate:0.01  
episode: 105   reward: -336.93459489380353  avg so far:-75.19882789836043 exploreRate:0.01  
episode: 106   reward: 229.78209717153396  avg so far:-72.29424765959955 exploreRate:0.01  
episode: 107   reward: -77.51533604138265  avg so far:-72.34350321037108 exploreRate:0.01  
episode: 108   reward: -71.89027911755697  avg so far:-72.3392674711859 exploreRate:0.01  
episode: 109   reward: 9.393330383748093  avg so far:-71.58248415771429 exploreRate:0.01  
episode: 110   reward: 7.341496780068098  avg so far:-70.8584109381016 exploreRate:0.01  
episode: 111   reward: -157.64585323736137  avg so far:-71.6473876862767 exploreRate:0.01  
episode: 112   reward: 198.420590902853  avg so far:-69.21434283412238 exploreRate:0.01  
episode: 113   reward: 196.8990041089059  avg so far:-66.83833080784534 exploreRate:0.01  
episode: 114   reward: -77.43492946532764  avg so far:-66.93210601720358 exploreRate:0.01  
episode: 115   reward: 3.935061363606451  avg so far:-66.31046419807367 exploreRate:0.01  
episode: 116   reward: -29.80371069665881  avg so far:-65.99301416762658 exploreRate:0.01  
episode: 117   reward: -51.81737308409937  avg so far:-65.87081036518238 exploreRate:0.01  
episode: 118   reward: -335.40806568936665  avg so far:-68.17454759017541 exploreRate:0.01  
episode: 119   reward: -15.657469559774675  avg so far:-67.72948760686693 exploreRate:0.01  
episode: 120   reward: 275.90033920472126  avg so far:-64.8418420034082 exploreRate:0.01  
episode: 121   reward: -8.106385112121075  avg so far:-64.36904652931415 exploreRate:0.01  
episode: 122   reward: 216.76025006153472  avg so far:-62.045663912860846 exploreRate:0.01  
episode: 123   reward: -1.4695334393046693  avg so far:-61.54913825324153 exploreRate:0.01  
episode: 124   reward: -43.09710161369794  avg so far:-61.39912169519646 exploreRate:0.01  
episode: 125   reward: -37.324038191650274  avg so far:-61.20496779597432 exploreRate:0.01  
episode: 126   reward: -69.47906154805904  avg so far:-61.271160545990995 exploreRate:0.01  
episode: 127   reward: 267.52066787368915  avg so far:-58.66170159027924 exploreRate:0.01  
episode: 128   reward: -46.66153241536894  avg so far:-58.56721206921696 exploreRate:0.01  
episode: 129   reward: -47.41841036086162  avg so far:-58.48011205587043 exploreRate:0.01  
episode: 130   reward: 249.02664068538203  avg so far:-56.09633877880646 exploreRate:0.01  
episode: 131   reward: 34.847613492433624  avg so far:-55.39676991518153 exploreRate:0.01  
episode: 132   reward: 249.52656293018734  avg so far:-53.06911088582757 exploreRate:0.01  
episode: 133   reward: -404.07223866339547  avg so far:-55.72822549020308 exploreRate:0.01  
episode: 134   reward: -28.56523260464428  avg so far:-55.52399246098835 exploreRate:0.01  
episode: 135   reward: 218.48126621603146  avg so far:-53.47917709772701 exploreRate:0.01  
episode: 136   reward: -275.136384106125  avg so far:-55.121082334826255 exploreRate:0.01  
episode: 137   reward: -112.95062299301904  avg so far:-55.546299545548266 exploreRate:0.01  
episode: 138   reward: 7.831569389439409  avg so far:-55.08368736354105 exploreRate:0.01  
episode: 139   reward: -19.76979524588242  avg so far:-54.82778959457251 exploreRate:0.01  
episode: 140   reward: -596.9429072142426  avg so far:-58.72789835442625 exploreRate:0.01  
episode: 141   reward: -373.91560164250524  avg so far:-60.97923909219825 exploreRate:0.01  
episode: 142   reward: -561.6197271346612  avg so far:-64.52988085136465 exploreRate:0.01  
episode: 143   reward: -103.58509614997791  avg so far:-64.80491757881967 exploreRate:0.01  
episode: 144   reward: -222.8742601187646  avg so far:-65.91029759658153 exploreRate:0.01  
episode: 145   reward: -681.0385860006248  avg so far:-70.1820218216096 exploreRate:0.01  
episode: 146   reward: -72.1566940587703  avg so far:-70.1956402508314 exploreRate:0.01  
episode: 147   reward: -29.79129292294074  avg so far:-69.91889814584584 exploreRate:0.01  
episode: 148   reward: 201.13615130871852  avg so far:-68.07498624479439 exploreRate:0.01  
episode: 149   reward: -414.46013950073075  avg so far:-70.41542646949667 exploreRate:0.01  
episode: 150   reward: -320.3842190682818  avg so far:-72.09306937284423 exploreRate:0.01  
episode: 151   reward: 200.36070766421315  avg so far:-70.27671085926384 exploreRate:0.01  
episode: 152   reward: -65.94109425162843  avg so far:-70.24799816649804 exploreRate:0.01  
episode: 153   reward: -100.50807329279463  avg so far:-70.4470776081184 exploreRate:0.01  
episode: 154   reward: -52.71018235337198  avg so far:-70.33115018815275 exploreRate:0.01  
episode: 155   reward: -70.7745098032415  avg so far:-70.33402914669229 exploreRate:0.01  
episode: 156   reward: -53.956772930756216  avg so far:-70.22836942917012 exploreRate:0.01  
episode: 157   reward: 224.01605538334638  avg so far:-68.34218721883347 exploreRate:0.01  
episode: 158   reward: 208.2362241631032  avg so far:-66.58054128646444 exploreRate:0.01  
episode: 159   reward: 136.88240890826702  avg so far:-65.29280109535856 exploreRate:0.01  
episode: 160   reward: -5.739446117453227  avg so far:-64.9182516929818 exploreRate:0.01  
episode: 161   reward: -324.10801296854004  avg so far:-66.53818770095403 exploreRate:0.01  
episode: 162   reward: -81.8440701377666  avg so far:-66.63325529372926 exploreRate:0.01  
episode: 163   reward: 212.05408233346049  avg so far:-64.91296308615401 exploreRate:0.01  
episode: 164   reward: 267.9580520187974  avg so far:-62.87080961925246 exploreRate:0.01  
episode: 165   reward: -55.041190962239625  avg so far:-62.82306804207556 exploreRate:0.01  
episode: 166   reward: 13.237372789318613  avg so far:-62.36209567340044 exploreRate:0.01  
episode: 167   reward: 216.9954926433739  avg so far:-60.679218635347574 exploreRate:0.01  
episode: 168   reward: -55.65583621453549  avg so far:-60.649138501091215 exploreRate:0.01  
episode: 169   reward: 162.912944523963  avg so far:-59.31841181641827 exploreRate:0.01  
episode: 170   reward: 254.753032344092  avg so far:-57.46000090422591 exploreRate:0.01  
episode: 171   reward: -75.02932149243166  avg so far:-57.56334984886241 exploreRate:0.01  
episode: 172   reward: -282.9065350059629  avg so far:-58.8811462532899 exploreRate:0.01  
episode: 173   reward: 213.71724183625787  avg so far:-57.296271903932066 exploreRate:0.01  
episode: 174   reward: 266.55773808438295  avg so far:-55.424283406889785 exploreRate:0.01  
episode: 175   reward: 225.42922792967155  avg so far:-53.81018276702449 exploreRate:0.01  
episode: 176   reward: 192.76370385777662  avg so far:-52.401189129168486 exploreRate:0.01  
episode: 177   reward: 104.21345336244012  avg so far:-51.511333205920714 exploreRate:0.01  
episode: 178   reward: 29.91544104689573  avg so far:-51.05129493330593 exploreRate:0.01  
episode: 179   reward: 257.127358374307  avg so far:-49.31995418438675 exploreRate:0.01  
episode: 180   reward: 276.42144259947366  avg so far:-47.50016984481211 exploreRate:0.01  
episode: 181   reward: 257.76834759652206  avg so far:-45.804233636804696 exploreRate:0.01  
episode: 182   reward: 31.16572628903474  avg so far:-45.37898523942437 exploreRate:0.01  
episode: 183   reward: 225.32582594584682  avg so far:-43.89159616697782 exploreRate:0.01  
episode: 184   reward: -98.01051817014667  avg so far:-44.187327981202785 exploreRate:0.01  
episode: 185   reward: 274.5143616356618  avg so far:-42.45525358111113 exploreRate:0.01  
episode: 186   reward: -136.17144422171037  avg so far:-42.96182758457383 exploreRate:0.01  
episode: 187   reward: 276.21540464602595  avg so far:-41.245820959678134 exploreRate:0.01  
episode: 188   reward: 245.19628848502418  avg so far:-39.71404497334282 exploreRate:0.01  
episode: 189   reward: 243.07391947810862  avg so far:-38.20985367306914 exploreRate:0.01  
episode: 190   reward: 250.69775723264772  avg so far:-36.681241975155295 exploreRate:0.01  
episode: 191   reward: 169.66571897320824  avg so far:-35.59520533858496 exploreRate:0.01  
episode: 192   reward: -11.558316915126696  avg so far:-35.46935775521607 exploreRate:0.01  
episode: 193   reward: -38.487251198601015  avg so far:-35.4850759502337 exploreRate:0.01  
episode: 194   reward: 261.6131314367452  avg so far:-33.945707000042106 exploreRate:0.01  
episode: 195   reward: 88.68414597562632  avg so far:-33.31359435583763 exploreRate:0.01  
episode: 196   reward: 246.26444312937312  avg so far:-31.879860830272445 exploreRate:0.01  
episode: 197   reward: 251.92505409186438  avg so far:-30.431876570465622 exploreRate:0.01  
episode: 198   reward: 231.71571436577486  avg so far:-29.101178139317195 exploreRate:0.01  
episode: 199   reward: 272.2123657171065  avg so far:-27.57939256428475 exploreRate:0.01  
episode: 200   reward: 222.4710808984072  avg so far:-26.32285752175866 exploreRate:0.01     
#### Sample Results: 
![Average Rewards](https://github.com/MohammadAsadolahi/DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python/blob/main/Results/Average%20Rewards.png)
![Total Rewards](https://github.com/MohammadAsadolahi/DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python/blob/main/Results/Total%20Rewards.png)
