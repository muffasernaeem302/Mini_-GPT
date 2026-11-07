import random
random.seed(42)

names = ["Ali","Sara","Hamza","Ayesha","Bilal","Zainab","Omar","Hira","Usman","Maryam","Daniyal","Fatima","Kamran","Noor","Imran","Sana"]
places = ["the old market","the quiet library","the river bank","the busy station","the mountain road","the small village","the city park","the tea shop","the school garden","the night bazaar"]
things = ["a red kite","an old map","a silver key","a broken clock","a warm blanket","a heavy book","a small lantern","a wooden box","a blue umbrella","a torn letter"]
feelings = ["happy","tired","curious","nervous","proud","calm","excited","worried"]
times = ["early in the morning","late in the evening","on a rainy afternoon","just before sunrise","on a cold winter night","during the summer holidays"]
verbs_find = ["found","noticed","picked up","carried","lost","kept"]
helpers = ["an old teacher","a kind shopkeeper","a young boy","a quiet neighbor","a wise grandmother","a friendly stranger"]
lessons = ["Patience is better than hurry.","Small kindness changes a long day.","A lost thing often leads to a new friend.","Hard work always finds its way.","Honesty is the best companion.","Every road teaches something."]
weather = ["The sky was clear.","It was raining softly.","A cool wind was blowing.","The sun was very bright.","The clouds were low and grey."]

def story():
    a, b = random.sample(names, 2)
    p, p2 = random.sample(places, 2)
    t = random.choice(things)
    f1, f2 = random.sample(feelings, 2)
    s = []
    s.append(f"{a} was walking to {p} {random.choice(times)}. {random.choice(weather)}")
    s.append(f"{a} felt {f1} and {random.choice(verbs_find)} {t} near the path.")
    s.append(f"\"What is this?\" said {a}. \"I must show it to someone.\"")
    s.append(f"At {p2}, {a} met {b} and {random.choice(helpers)}.")
    s.append(f"{b} looked at {t} and said, \"I have seen this before. It belongs to {random.choice(names)}.\"")
    s.append(f"Together they walked back and returned it. {random.choice(names)} smiled and felt {f2}.")
    s.append(f"{a} learned one thing that day: {random.choice(lessons)}")
    return "\n".join(s)

parts = []
size = 0
i = 1
while size < 600_000:
    st = f"STORY {i}\n{story()}\n"
    parts.append(st)
    size += len(st)
    i += 1
open("data/my_stories.txt","w",encoding="utf-8").write("\n".join(parts))
print(i-1, "stories,", size, "chars")
