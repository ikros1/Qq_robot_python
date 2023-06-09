from aicore import robot_talk_front_prompt, internet_analyze_front_prompt


class memory:
    def __init__(self):
        self.personal_memory = None
        self.group_memory = {}
        self.talk_memory = None
        self.temp_memory = None
        self.all_txt = []
        self.from_person = None
        self.send_flag = False
        self.authority = {}

    def init_ikaros_memory(self, form_group):
        ikaros_memory_site = {"ikaros_id": [], "group_memory": [], "assistant_id": [], "assistant_memory": []}
        for data in robot_talk_front_prompt("风纪委员"):
            ikaros_memory_site["ikaros_id"].append(data)
        for data in internet_analyze_front_prompt():
            ikaros_memory_site["assistant_id"].append(data)
        self.group_memory[form_group] = ikaros_memory_site

    def reload_robot_memory(self, form_group, robot_name):
        ikaros_memory_site = {"ikaros_id": [], "group_memory": [], "assistant_id": [], "assistant_memory": []}
        for data in robot_talk_front_prompt(robot_name):
            ikaros_memory_site["ikaros_id"].append(data)
        for data in internet_analyze_front_prompt():
            ikaros_memory_site["assistant_id"].append(data)
        self.group_memory[form_group] = ikaros_memory_site

    def init_authority(self, person):
        self.authority[person] = {"use_ai": True, "use_ai_voice": False, "use_ai_web_search": False}

    def admin_authority(self, person):
        self.authority[person] = {"use_ai": True, "use_ai_voice": True, "use_ai_web_search": True}

    def close_authority(self, person):
        self.authority[person] = {"use_ai": False, "use_ai_voice": False, "use_ai_web_search": False}

    def change_authority(self, person, key, value):
        self.authority[person][key] = value

    def clear_memory(self, form_group):
        # 检测self.group_memory是否超过30条，如果超过20在while中清除{"ikaros_id": [], "group_memory": []}中key为“group_memory”最早的记录,并且初始化ikaros记忆
        while len(self.group_memory[form_group]["group_memory"]) > 20:
            self.group_memory[form_group]["group_memory"].pop(0)
        while len(self.group_memory[form_group]["assistant_memory"]) > 3:
            self.group_memory[form_group]["assistant_memory"] = []
