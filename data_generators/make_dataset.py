"""
Dataset generator: chain, fork and collider stories.

Run:  python make_dataset.py        -> writes stories_chain.csv, stories_fork.csv, stories_collider.csv

Design notes
- Every scenario is NEW: none of the 20 pilot scenarios (rain/road, fire/smoke, flu/fever, power cut, car battery,
  cough, fire alarm, traffic jam, baby crying, dog barking, ice cream, fuel tank, exam score, shaking glasses,
  red skin, phone dead, crowd cheer, muddy shoes, blicket, zorp) is reused.
- Each scenario is expanded into ~10-13 surface variants (character name x neutral opener sentence).
  The statistical unit is the SCENARIO, not the variant: bootstrap by scenario_id.
- For every text we store the character span of the key sentence (mediator / cause), so token positions can be
  found later with the tokenizer's offset mapping (model-specific, so not stored here).

Chain    A -> B -> C   texts: base, see (B true), neg (B false), do (B forced by outside agent),
                       cause_t / cause_f (A stated true / false; for the H2 cause direction)
                       probes: q_cause (A), q_eff (C), q_med (B, manipulation check)
Fork     A -> B, A -> C  same texts; probes: q_cause (A), q_sib (C, the sibling effect), q_med (B)
Collider A -> C <- B   texts: base, see (C true), see_A (C true + A true), A_only, do (C forced), neg (C false),
                       b_t / b_f (B stated true / false)
                       probes: q_b (B, main), q_a (A), q_c (C)
"""
import csv, random

NAMES = ["Asha", "Rohan", "Meena", "Arjun", "Sara", "David", "Kavya", "Imran", "Lena", "Vikram",
         "Nisha", "Tom", "Ananya", "Karthik", "Maria", "Sameer"]
OPENERS = ["It was a Monday.", "It was a Tuesday in March.", "This happened last year.",
           "This story takes place in a small town.", "It was the middle of the week.",
           "It was a Thursday.", "This happened a few weeks ago.", "This story takes place in a big city.",
           "It was a Saturday.", "It was an ordinary day.", "It was a Wednesday in October.",
           "This happened some time ago.", "It was a Friday."]

# ---------------------------------------------------------------- CHAIN  A -> B -> C
CHAIN = [
 dict(id="ch_ice_bus", ctx="{n} was waiting for the school bus early in the morning.", slot="{n} looked at the road.",
      see="The road was covered in ice.", neg="The road was completely clear and dry.",
      do="A broken water pipe had flooded the road overnight and the water froze, so the road was covered in ice.",
      cause_t="It had snowed heavily during the night.", cause_f="It had not snowed at all during the night.",
      q_cause="Did it snow during the night?", q_eff="Was the bus late?", q_med="Was the road icy?"),
 dict(id="ch_tree_route", ctx="{n} was driving home along a narrow country lane.", slot="{n} looked ahead at the lane.",
      see="A large tree had fallen across the lane.", neg="The lane ahead was completely clear.",
      do="Workers had cut down a large tree for a building project, so it had fallen across the lane.",
      cause_t="There had been a violent storm that afternoon.", cause_f="The weather had been calm all day.",
      q_cause="Was there a storm?", q_eff="Did {n} have to take another route?", q_med="Was a tree lying across the lane?"),
 dict(id="ch_river_fields", ctx="{n} owns some farm fields next to the river.", slot="{n} walked down to look at the river.",
      see="The river had flooded its banks.", neg="The river was at its normal level.",
      do="Engineers had opened the gates of the dam upstream, so the river had flooded its banks.",
      cause_t="It had rained heavily for three days.", cause_f="It had not rained for weeks.",
      q_cause="Did it rain heavily?", q_eff="Were the fields underwater?", q_med="Had the river flooded?"),
 dict(id="ch_vomit_rest", ctx="{n} came home in the evening after going out with friends.", slot="{n} sat down in the living room.",
      see="{n} was vomiting.", neg="{n} felt perfectly fine.",
      do="{n} had just ridden a very fast roller coaster, so {n} was vomiting.",
      cause_t="{n} had eaten some bad food at lunch.", cause_f="{n} had eaten only fresh, safe food all day.",
      q_cause="Did {n} eat bad food?", q_eff="Did {n} need to lie down?", q_med="Was {n} vomiting?"),
 dict(id="ch_window_cold", ctx="It was a cold night, and {n} had gone out for dinner.", slot="When {n} came back, {n} checked the front room window.",
      see="The window was broken.", neg="The window was whole and closed.",
      do="{n}'s son had kicked a football through the window, so it was broken.",
      cause_t="A thief had broken into the house.", cause_f="Nobody had come near the house.",
      q_cause="Did a thief break into the house?", q_eff="Did the front room get cold?", q_med="Was the window broken?"),
 dict(id="ch_crops_income", ctx="{n} is a farmer who grows wheat.", slot="In July, {n} looked at the wheat fields.",
      see="The crops had dried up.", neg="The crops were green and healthy.",
      do="A worker had sprayed the wrong chemical on the fields by mistake, so the crops had dried up.",
      cause_t="There had been a long drought.", cause_f="There had been plenty of rain that season.",
      q_cause="Was there a drought?", q_eff="Did {n} earn less money that year?", q_med="Had the crops dried up?"),
 dict(id="ch_ash_flights", ctx="{n} was at the airport, about to fly to another city.", slot="{n} looked out of the window at the sky.",
      see="The sky was full of thick ash.", neg="The sky was clear and blue.",
      do="A huge factory fire nearby had filled the sky with thick ash.",
      cause_t="A volcano had erupted nearby.", cause_f="There are no active volcanoes anywhere near the airport.",
      q_cause="Did a volcano erupt?", q_eff="Were flights cancelled?", q_med="Was there ash in the sky?"),
 dict(id="ch_hand_write", ctx="{n} had an important written exam in the afternoon.", slot="Before the exam, {n} looked at {n}'s right hand.",
      see="{n}'s hand was badly swollen.", neg="{n}'s hand looked completely normal.",
      do="{n} had slammed the hand in a car door, so it was badly swollen.",
      cause_t="A bee had stung {n} on the hand.", cause_f="{n} had stayed indoors all morning, away from any insects.",
      q_cause="Was {n} stung by a bee?", q_eff="Was it hard for {n} to write?", q_med="Was {n}'s hand swollen?"),
 dict(id="ch_sea_swim", ctx="{n} went to the beach for a swim.", slot="{n} looked at the water near the beach.",
      see="The water was black and dirty.", neg="The water was clean and clear.",
      do="A factory had dumped black dye into the sea, so the water was black and dirty.",
      cause_t="An oil tanker had spilled oil near the coast.", cause_f="No ships had passed near the coast for weeks.",
      q_cause="Was there an oil spill?", q_eff="Did people stop swimming there?", q_med="Was the water black?"),
 dict(id="ch_tired_mistakes", ctx="{n} works as an accountant in an office.", slot="{n} arrived at work in the morning.",
      see="{n} was extremely tired.", neg="{n} was fresh and full of energy.",
      do="{n} had run a full marathon early that morning, so {n} was extremely tired.",
      cause_t="{n} had stayed up until four in the morning.", cause_f="{n} had slept a full nine hours.",
      q_cause="Did {n} stay up late the night before?", q_eff="Did {n} make mistakes at work?", q_med="Was {n} tired?"),
 dict(id="ch_beam_floor", ctx="{n} lives in an old wooden house.", slot="{n} went down to look at the main beam under the floor.",
      see="The main beam was weak and crumbling.", neg="The main beam was solid and strong.",
      do="A builder had sawn halfway through the beam by mistake, so it was weak.",
      cause_t="Termites had been eating the wood for years.", cause_f="The house had been checked and was free of insects.",
      q_cause="Were there termites in the house?", q_eff="Did the floor start to sag?", q_med="Was the beam weak?"),
 dict(id="ch_stain_repair", ctx="{n} lives on the top floor of a house.", slot="{n} looked up at the bedroom ceiling.",
      see="There was a big brown water stain on the ceiling.", neg="The ceiling was clean and white.",
      do="{n}'s child had thrown a cup of tea at the ceiling, so there was a big brown stain on it.",
      cause_t="The roof had been leaking for weeks.", cause_f="The roof had just been replaced and was watertight.",
      q_cause="Was the roof leaking?", q_eff="Did {n} call someone to fix it?", q_med="Was there a stain on the ceiling?"),
 dict(id="ch_spots_notice", ctx="{n}'s daughter went to school in the morning.", slot="Her teacher looked at her face.",
      see="Her face was covered in red spots.", neg="Her face was completely clear.",
      do="Her little brother had drawn red spots on her face with a marker, so her face was covered in red spots.",
      cause_t="She had caught measles.", cause_f="She had been vaccinated and was completely healthy.",
      q_cause="Did the girl have measles?", q_eff="Did the other children notice the spots?", q_med="Did the girl have red spots on her face?"),
 dict(id="ch_smell_leave", ctx="{n} works in a large office building.", slot="{n} walked into the hallway.",
      see="There was a strong smell of gas in the hallway.", neg="The hallway smelled completely normal.",
      do="A prankster had sprayed a bottle of gas-smelling chemical in the hallway, so it smelled strongly of gas.",
      cause_t="A gas pipe in the building was leaking.", cause_f="The building has no gas pipes at all.",
      q_cause="Was there a gas leak?", q_eff="Did people leave the building?", q_med="Did the hallway smell of gas?"),
 dict(id="ch_dents_repair", ctx="{n} had left the car parked outside overnight.", slot="In the morning, {n} looked at the car.",
      see="The car was covered in small dents.", neg="The car had no marks on it at all.",
      do="A neighbour's child had thrown stones at the car, so it was covered in small dents.",
      cause_t="There had been a hailstorm during the night.", cause_f="The night had been calm and clear.",
      q_cause="Was there a hailstorm?", q_eff="Did the car need repairs?", q_med="Was the car dented?"),
 dict(id="ch_chickens_eggs", ctx="{n} keeps chickens on a small farm.", slot="In the morning, {n} counted the chickens.",
      see="Three chickens were missing.", neg="All of the chickens were there.",
      do="{n}'s son had sold three chickens at the market, so three were missing.",
      cause_t="A fox had got into the farm during the night.", cause_f="The farm fence was strong and no animals had got in.",
      q_cause="Did a fox get into the farm?", q_eff="Did {n} collect fewer eggs?", q_med="Were some chickens missing?"),
 dict(id="ch_tyre_walk", ctx="{n} cycled to the shops and left the bicycle outside.", slot="When {n} came out, {n} looked at the bicycle.",
      see="The back tyre was flat.", neg="Both tyres were fully pumped up.",
      do="A prankster had let the air out of the back tyre, so it was flat.",
      cause_t="The bicycle had run over a nail on the way.", cause_f="The road had been perfectly clean the whole way.",
      q_cause="Did the bicycle run over a nail?", q_eff="Did {n} have to walk the bicycle home?", q_med="Was the tyre flat?"),
 dict(id="ch_lake_fish", ctx="{n} likes fishing at the lake on weekends.", slot="{n} looked at the lake.",
      see="The water level in the lake was very low.", neg="The lake was full of water.",
      do="Workers had drained the lake to repair the dam, so the water level was very low.",
      cause_t="There had been no rain for months.", cause_f="It had rained a lot all season.",
      q_cause="Had there been no rain for months?", q_eff="Did {n} catch fewer fish?", q_med="Was the water level low?"),
 dict(id="ch_voice_sing", ctx="{n} was going to sing at a concert that evening.", slot="{n} tried to speak in the afternoon.",
      see="{n}'s voice was hoarse.", neg="{n}'s voice was clear and strong.",
      do="{n} had shouted for three hours at a football match, so {n}'s voice was hoarse.",
      cause_t="{n} had caught a throat infection.", cause_f="{n} was completely healthy.",
      q_cause="Did {n} have a throat infection?", q_eff="Did {n} sing badly at the concert?", q_med="Was {n}'s voice hoarse?"),
 dict(id="ch_cracks_leave", ctx="{n} lives in an apartment building.", slot="{n} looked at the walls of the building.",
      see="There were big new cracks in the walls.", neg="The walls had no cracks at all.",
      do="Workers drilling next door had damaged the walls, so there were big new cracks in them.",
      cause_t="An earthquake had struck the city.", cause_f="There had been no earthquake.",
      q_cause="Was there an earthquake?", q_eff="Did the residents worry about safety?", q_med="Were there cracks in the walls?"),
]

# ---------------------------------------------------------------- FORK  A -> B, A -> C
FORK = [
 dict(id="fk_branches_furniture", ctx="{n} has a garden with a table and chairs.", slot="In the morning, {n} looked at the lawn.",
      see="There were broken branches all over the lawn.", neg="The lawn was neat and clear.",
      do="A gardener had cut some branches and left them all over the lawn.",
      cause_t="A strong storm had passed during the night.", cause_f="The night had been completely calm.",
      q_cause="Was there a storm?", q_sib="Were the garden chairs knocked over?", q_med="Were there branches on the lawn?"),
 dict(id="fk_music_cars", ctx="{n} lives on a quiet street.", slot="That evening, {n} listened to the house next door.",
      see="Very loud music was coming from next door.", neg="It was completely quiet next door.",
      do="The neighbour was testing new speakers alone, so very loud music was coming from next door.",
      cause_t="The neighbours were having a big party.", cause_f="The neighbours were away on holiday.",
      q_cause="Were the neighbours having a party?", q_sib="Were lots of cars parked on the street?", q_med="Was there loud music?"),
 dict(id="fk_bread_queue", ctx="{n} walked past the bakery at six in the morning.", slot="{n} noticed the air on the street.",
      see="There was a strong smell of fresh bread.", neg="There was no smell of bread at all.",
      do="A candle shop was testing a bread-scented candle, so there was a strong smell of fresh bread.",
      cause_t="The bakery had opened early that day.", cause_f="The bakery was closed that day.",
      q_cause="Was the bakery open?", q_sib="Was there a queue outside the bakery?", q_med="Did it smell of bread?"),
 dict(id="fk_park_school", ctx="{n} walked through the park at ten in the morning on a weekday.", slot="{n} looked around the park.",
      see="The park was full of children.", neg="There were no children in the park.",
      do="A birthday party had brought many children to the park, so it was full of children.",
      cause_t="It was a school holiday.", cause_f="It was a normal school day.",
      q_cause="Was it a school holiday?", q_sib="Was the school building empty?", q_med="Was the park full of children?"),
 dict(id="fk_lift_lights", ctx="{n} works on the tenth floor of an office building.", slot="{n} pressed the button for the lift.",
      see="The lift had stopped working.", neg="The lift was working normally.",
      do="A technician had switched off the lift for maintenance, so it had stopped working.",
      cause_t="There was a power cut in the building.", cause_f="The building's power supply was working normally.",
      q_cause="Was there a power cut?", q_sib="Were the corridor lights off?", q_med="Had the lift stopped working?"),
 dict(id="fk_pond_coats", ctx="{n} went for a walk in the town park.", slot="{n} looked at the pond.",
      see="The pond was frozen solid.", neg="The pond was not frozen at all.",
      do="Park staff had used freezing machines for an ice show, so the pond was frozen solid.",
      cause_t="The weather was bitterly cold.", cause_f="The weather was warm and sunny.",
      q_cause="Was the weather very cold?", q_sib="Were people wearing thick coats?", q_med="Was the pond frozen?"),
 dict(id="fk_stomach_guests", ctx="{n} went to a big dinner with twenty guests.", slot="Later that night, {n} noticed how {n}'s stomach felt.",
      see="{n} had a bad stomach ache.", neg="{n}'s stomach felt fine.",
      do="{n} had done two hundred sit-ups after dinner, so {n} had a bad stomach ache.",
      cause_t="The food at the dinner had gone bad.", cause_f="The food at the dinner was fresh and safe.",
      q_cause="Was the food at the dinner bad?", q_sib="Did other guests also get sick?", q_med="Did {n} have a stomach ache?"),
 dict(id="fk_siren_glass", ctx="{n} was sitting in a cafe next to a busy road.", slot="{n} listened to the street outside.",
      see="An ambulance siren was wailing outside.", neg="The street outside was quiet.",
      do="An ambulance crew was practising for a training drill, so a siren was wailing outside.",
      cause_t="There had been a car crash on the road.", cause_f="There had been no accidents on the road.",
      q_cause="Was there a car crash?", q_sib="Was there broken glass on the road?", q_med="Was there a siren?"),
 dict(id="fk_pictures_news", ctx="{n} came home in the afternoon.", slot="{n} looked at the living room walls.",
      see="All the pictures had fallen off the walls.", neg="All the pictures were still hanging on the walls.",
      do="{n}'s children had been playing football indoors and knocked all the pictures off the walls.",
      cause_t="An earthquake had shaken the town.", cause_f="There had been no earthquake.",
      q_cause="Was there an earthquake?", q_sib="Did the news report shaking in the town?", q_med="Had the pictures fallen off the walls?"),
 dict(id="fk_car_quit", ctx="{n} had worked at the same factory for many years.", slot="One Monday, {n} arrived in the car park.",
      see="{n} was driving an expensive new car.", neg="{n} was driving the same old car as always.",
      do="The company had given {n} an expensive new car as a work vehicle, so {n} was driving it.",
      cause_t="{n} had won the lottery.", cause_f="{n} had never bought a lottery ticket.",
      q_cause="Did {n} win the lottery?", q_sib="Did {n} quit the job soon after?", q_med="Did {n} have a new car?"),
 dict(id="fk_tar_pool", ctx="{n} walked to the shops in the afternoon.", slot="{n} looked down at the road.",
      see="The tar on the road had gone soft and sticky.", neg="The road surface was hard and normal.",
      do="Road workers had heated the tar with machines, so it had gone soft and sticky.",
      cause_t="There was a heatwave.", cause_f="The weather was cool and rainy.",
      q_cause="Was there a heatwave?", q_sib="Was the swimming pool crowded?", q_med="Was the tar soft?"),
 dict(id="fk_tired_babyclothes", ctx="{n} visited {n}'s friends, a young couple.", slot="{n} looked at the couple.",
      see="The couple looked extremely tired.", neg="The couple looked well rested.",
      do="The couple had been painting their house all night, so they looked extremely tired.",
      cause_t="The couple had just had a new baby.", cause_f="The couple do not have any children.",
      q_cause="Did the couple have a new baby?", q_sib="Had the couple bought baby clothes?", q_med="Did the couple look tired?"),
 dict(id="fk_fans_trophy", ctx="{n} lives in the city where the local football team plays.", slot="That evening, {n} looked out at the main street.",
      see="Crowds of fans were celebrating in the street.", neg="The main street was empty and quiet.",
      do="A film crew had paid crowds of people to celebrate in the street for a movie scene.",
      cause_t="The local team had won the cup final.", cause_f="The local team had lost the cup final.",
      q_cause="Did the local team win the final?", q_sib="Was the trophy shown on the evening news?", q_med="Were fans celebrating in the street?"),
 dict(id="fk_call_neighbours", ctx="{n} was on a video call for work at home.", slot="{n} looked at the laptop screen.",
      see="The video call had suddenly dropped.", neg="The video call was running smoothly.",
      do="{n}'s little sister had unplugged the laptop, so the video call had dropped.",
      cause_t="The internet had gone down across the whole area.", cause_f="The internet in the area was working perfectly.",
      q_cause="Did the internet go down in the area?", q_sib="Did the neighbours also lose their internet?", q_med="Did the video call drop?"),
 dict(id="fk_busstop_roads", ctx="{n} went to the bus stop at eight in the morning.", slot="{n} looked at the bus stop.",
      see="The bus stop was extremely crowded.", neg="The bus stop was almost empty.",
      do="A free concert nearby had drawn a big crowd to the bus stop.",
      cause_t="The bus drivers were on strike.", cause_f="All buses were running normally.",
      q_cause="Were the bus drivers on strike?", q_sib="Were the roads busier with cars than usual?", q_med="Was the bus stop crowded?"),
]

# ---------------------------------------------------------------- COLLIDER  A -> C <- B
# rule names both causes; B is the cause we probe; A is the cause that explains away
COLLIDER = [
 dict(id="co_grass", ctx="In {n}'s garden, the grass gets wet if it rains or if the sprinkler is turned on.", slot="In the morning, {n} looked at the grass.",
      c_see="The grass was wet.", c_neg="The grass was completely dry.", c_do="A neighbour had poured buckets of water on the grass, so it was wet.",
      a_t="The sprinkler had been turned on during the night.", b_t="It had rained during the night.", b_f="It had not rained during the night.",
      q_b="Did it rain during the night?", q_a="Was the sprinkler turned on?", q_c="Was the grass wet?"),
 dict(id="co_alarm", ctx="The alarm in {n}'s house goes off if a burglar breaks in or if the cat jumps onto the sensor.", slot="{n} was out at work.",
      c_see="The house alarm went off.", c_neg="The house alarm stayed silent all day.", c_do="A security company tested the alarm, so it went off.",
      a_t="The cat had jumped onto the sensor.", b_t="A burglar had broken in.", b_f="No burglar had come near the house.",
      q_b="Did a burglar break in?", q_a="Did the cat jump onto the sensor?", q_c="Did the alarm go off?"),
 dict(id="co_flight", ctx="{n}'s flight is delayed if there is a storm or if there is an engine problem.", slot="{n} checked the departures board.",
      c_see="The flight was delayed.", c_neg="The flight was on time.", c_do="The airline delayed the flight on purpose to wait for a VIP passenger.",
      a_t="Mechanics had found an engine problem.", b_t="There was a storm.", b_f="The weather was calm and clear.",
      q_b="Was there a storm?", q_a="Was there an engine problem?", q_c="Was the flight delayed?"),
 dict(id="co_headache", ctx="{n} gets a headache when {n} does not sleep enough or when {n} skips meals.", slot="{n} went to work.",
      c_see="{n} had a headache.", c_neg="{n} felt completely fine.", c_do="{n} was hit on the head by a falling box, so {n} had a headache.",
      a_t="{n} had skipped breakfast and lunch.", b_t="{n} had slept only three hours.", b_f="{n} had slept a full nine hours.",
      q_b="Did {n} sleep badly?", q_a="Did {n} skip meals?", q_c="Did {n} have a headache?"),
 dict(id="co_plant", ctx="{n}'s plant wilts if it is not watered or if it gets too much sun.", slot="{n} looked at the plant.",
      c_see="The plant had wilted.", c_neg="The plant was healthy and upright.", c_do="{n}'s child had bent the plant's stems, so it looked wilted.",
      a_t="The plant had been left in strong sun all day.", b_t="Nobody had watered the plant for a week.", b_f="The plant had been watered every day.",
      q_b="Had the plant gone without water?", q_a="Had the plant been in strong sun?", q_c="Had the plant wilted?"),
 dict(id="co_floor", ctx="The kitchen floor in {n}'s house gets wet if the roof leaks or if someone spills water.", slot="{n} walked into the kitchen.",
      c_see="The kitchen floor was wet.", c_neg="The kitchen floor was dry.", c_do="A cleaner had just mopped the floor, so it was wet.",
      a_t="{n}'s son had spilled a bucket of water.", b_t="The roof had been leaking.", b_f="The roof was in perfect condition.",
      q_b="Was the roof leaking?", q_a="Did someone spill water?", q_c="Was the kitchen floor wet?"),
 dict(id="co_train", ctx="{n}'s train runs late if there is a signal failure or if there is heavy snow.", slot="{n} waited on the platform.",
      c_see="The train was running late.", c_neg="The train arrived on time.", c_do="The driver stopped the train on purpose to film a video, so it was late.",
      a_t="There was heavy snow on the tracks.", b_t="There had been a signal failure.", b_f="All the signals were working perfectly.",
      q_b="Was there a signal failure?", q_a="Was there heavy snow?", q_c="Was the train late?"),
 dict(id="co_website", ctx="{n}'s company website goes down if the server crashes or if there is planned maintenance.", slot="{n} opened the website.",
      c_see="The website was down.", c_neg="The website was working normally.", c_do="A worker had pulled out the network cable by accident, so the website was down.",
      a_t="There was planned maintenance that day.", b_t="The server had crashed.", b_f="The server was running perfectly.",
      q_b="Did the server crash?", q_a="Was there planned maintenance?", q_c="Was the website down?"),
 dict(id="co_shop", ctx="The corner shop near {n}'s house closes if it is a public holiday or if the owner is ill.", slot="{n} walked to the corner shop.",
      c_see="The shop was closed.", c_neg="The shop was open.", c_do="The police had closed the shop for a few hours to film a TV show.",
      a_t="The owner was ill in bed.", b_t="It was a public holiday.", b_f="It was an ordinary working day.",
      q_b="Was it a public holiday?", q_a="Was the owner ill?", q_c="Was the shop closed?"),
 dict(id="co_road", ctx="The main road to {n}'s village is blocked if there is an accident or if there is construction work.", slot="{n} drove towards the village.",
      c_see="The main road was blocked.", c_neg="The main road was open.", c_do="Protesters had parked their tractors across the road, so it was blocked.",
      a_t="There was construction work on the road.", b_t="There had been an accident.", b_f="There had been no accidents.",
      q_b="Was there an accident?", q_a="Was there construction work?", q_c="Was the road blocked?"),
 dict(id="co_milk", ctx="Milk goes sour if it is left out of the fridge or if it is past its expiry date.", slot="{n} opened the milk.",
      c_see="The milk was sour.", c_neg="The milk was fresh.", c_do="{n}'s brother had added lemon juice to the milk as a joke, so it was sour.",
      a_t="The milk was past its expiry date.", b_t="The milk had been left out of the fridge.", b_f="The milk had stayed in the fridge.",
      q_b="Was the milk left out of the fridge?", q_a="Was the milk past its expiry date?", q_c="Was the milk sour?"),
 dict(id="co_screen", ctx="{n}'s phone screen cracks if the phone is dropped or if someone sits on it.", slot="{n} picked up the phone.",
      c_see="The phone screen was cracked.", c_neg="The phone screen was perfect.", c_do="A repair shop had cracked the screen while testing it.",
      a_t="Someone had sat on the phone.", b_t="{n} had dropped the phone.", b_f="{n} had never dropped the phone.",
      q_b="Did {n} drop the phone?", q_a="Did someone sit on the phone?", q_c="Was the screen cracked?"),
 dict(id="co_exam", ctx="Students in {n}'s class fail the exam if they do not study or if they are ill on exam day.", slot="{n} received the exam results.",
      c_see="{n} had failed the exam.", c_neg="{n} had passed the exam easily.", c_do="The teacher had entered the wrong marks by mistake, so {n} was listed as failed.",
      a_t="{n} had been ill on exam day.", b_t="{n} had not studied at all.", b_f="{n} had studied every day for weeks.",
      q_b="Did {n} skip studying?", q_a="Was {n} ill on exam day?", q_c="Did {n} fail the exam?"),
 dict(id="co_toast", ctx="Toast in {n}'s kitchen gets burnt if the toaster is faulty or if it is left in too long.", slot="{n} took the toast out.",
      c_see="The toast was burnt.", c_neg="The toast was perfectly golden.", c_do="{n}'s friend had burnt the toast with a lighter as a prank.",
      a_t="{n} had left the toast in for far too long.", b_t="The toaster was faulty.", b_f="The toaster was brand new and working perfectly.",
      q_b="Was the toaster faulty?", q_a="Was the toast left in too long?", q_c="Was the toast burnt?"),
 dict(id="co_match", ctx="The school football match is cancelled if there is heavy rain or if the referee is ill.", slot="{n} arrived at the football ground.",
      c_see="The match was cancelled.", c_neg="The match went ahead as planned.", c_do="The principal cancelled the match to use the ground for a ceremony.",
      a_t="The referee was ill.", b_t="There was heavy rain.", b_f="The weather was dry and sunny.",
      q_b="Was there heavy rain?", q_a="Was the referee ill?", q_c="Was the match cancelled?"),
 dict(id="co_bill", ctx="{n}'s electricity bill is very high if the air conditioner runs all month or if the meter is faulty.", slot="{n} opened the electricity bill.",
      c_see="The bill was very high.", c_neg="The bill was normal.", c_do="A clerk had typed an extra zero by mistake, so the bill was very high.",
      a_t="The electricity meter was faulty.", b_t="The air conditioner had run all month.", b_f="The air conditioner had not been used at all.",
      q_b="Did the air conditioner run all month?", q_a="Was the meter faulty?", q_c="Was the bill high?"),
 dict(id="co_race", ctx="Runners in {n}'s club lose races if they are injured or if they start late.", slot="{n} ran in the race.",
      c_see="{n} lost the race.", c_neg="{n} won the race.", c_do="The judges disqualified {n} by mistake, so {n} lost the race.",
      a_t="{n} had started late.", b_t="{n} had an injured ankle.", b_f="{n} was completely fit and uninjured.",
      q_b="Was {n} injured?", q_a="Did {n} start late?", q_c="Did {n} lose the race?"),
 dict(id="co_crying", ctx="{n}'s baby cries if he is hungry or if his nappy is wet.", slot="{n} listened at the baby's door.",
      c_see="The baby was crying.", c_neg="The baby was quiet and calm.", c_do="A loud firework outside had startled the baby, so he was crying.",
      a_t="The baby's nappy was wet.", b_t="The baby was hungry.", b_f="The baby had just been fed.",
      q_b="Was the baby hungry?", q_a="Was the baby's nappy wet?", q_c="Was the baby crying?"),
 dict(id="co_lights", ctx="The street lights on {n}'s road go out if there is a power cut or if the bulbs burn out.", slot="{n} walked home along the road at night.",
      c_see="The street lights were off.", c_neg="The street lights were on.", c_do="The council had switched the street lights off to save money.",
      a_t="The bulbs had burnt out.", b_t="There was a power cut.", b_f="The power was working normally.",
      q_b="Was there a power cut?", q_a="Had the bulbs burnt out?", q_c="Were the street lights off?"),
]


def fill(s, n):
    return s.replace("{n}", n)


def assemble(opener, ctx, *sents):
    """Join sentences; return text and char span of the LAST sentence (the key sentence)."""
    parts = [opener, ctx] + list(sents[:-1])
    head = " ".join(p for p in parts if p)
    text = head + " " + sents[-1]
    start = len(head) + 1
    return text, [start, start + len(sents[-1])]


def assemble_mid(opener, ctx, key, *after):
    """Key sentence followed by more sentences; return text and the key sentence span."""
    head = " ".join(p for p in [opener, ctx] if p)
    start = len(head) + 1
    text = head + " " + key + ("" if not after else " " + " ".join(after))
    return text, [start, start + len(key)]


def variants(k, seed):
    rng = random.Random(seed)
    pairs = [(o, n) for o in OPENERS for n in NAMES]
    rng.shuffle(pairs)
    return pairs[:k]


def build_chain_like(bank, structure, per, q2_key):
    rows = []
    for si, s in enumerate(bank):
        for vi, (op, n) in enumerate(variants(per, seed=hash(s["id"]) % 10**6)):
            f = {k: fill(v, n) for k, v in s.items()}
            r = dict(structure=structure, scenario_id=s["id"], item_id=f"{s['id']}_{vi}", name=n, opener=op)
            for cond, key in [("base", "slot"), ("see", "see"), ("neg", "neg"), ("do", "do")]:
                r[cond], r[f"span_{cond}"] = assemble(op, f["ctx"], f[key])
            for cond, key in [("cause_t", "cause_t"), ("cause_f", "cause_f")]:
                r[cond], r[f"span_{cond}"] = assemble(op, f["ctx"], f[key])
            r["q_cause"], r[q2_key], r["q_med"] = f["q_cause"], f[q2_key], f["q_med"]
            rows.append(r)
    return rows


def build_collider(bank, per):
    rows = []
    for s in bank:
        for vi, (op, n) in enumerate(variants(per, seed=hash(s["id"]) % 10**6)):
            f = {k: fill(v, n) for k, v in s.items()}
            r = dict(structure="collider", scenario_id=s["id"], item_id=f"{s['id']}_{vi}", name=n, opener=op)
            r["base"], r["span_base"] = assemble(op, f["ctx"], f["slot"])
            r["see"], r["span_see"] = assemble(op, f["ctx"], f["c_see"])
            r["neg"], r["span_neg"] = assemble(op, f["ctx"], f["c_neg"])
            r["do"], r["span_do"] = assemble(op, f["ctx"], f["c_do"])
            r["see_A"], r["span_see_A"] = assemble_mid(op, f["ctx"], f["c_see"], f["a_t"])    # span = C sentence
            r["A_only"], r["span_A_only"] = assemble_mid(op, f["ctx"], f["slot"], f["a_t"])   # span = slot sentence
            r["b_t"], r["span_b_t"] = assemble(op, f["ctx"], f["b_t"])
            r["b_f"], r["span_b_f"] = assemble(op, f["ctx"], f["b_f"])
            r["q_b"], r["q_a"], r["q_c"] = f["q_b"], f["q_a"], f["q_c"]
            rows.append(r)
    return rows


def write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, list) else v) for k, v in r.items()})


if __name__ == "__main__":
    # use a fixed hash so output is identical on every run
    import zlib
    hash = lambda s: zlib.crc32(s.encode())  # noqa: E731
    chain = build_chain_like(CHAIN, "chain", per=24, q2_key="q_eff")  # variants 0-15 learn directions, 16-23 held out
    fork = build_chain_like(FORK, "fork", per=13, q2_key="q_sib")
    coll = build_collider(COLLIDER, per=11)
    write(chain, "stories_chain.csv"); write(fork, "stories_fork.csv"); write(coll, "stories_collider.csv")
    print(f"chain: {len(chain)} items / {len(CHAIN)} scenarios")
    print(f"fork: {len(fork)} items / {len(FORK)} scenarios")
    print(f"collider: {len(coll)} items / {len(COLLIDER)} scenarios")
