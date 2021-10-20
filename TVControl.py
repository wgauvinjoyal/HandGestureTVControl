
#Class to control a tv
class TVControl:
    device_id = ""
    device_name = ""
    IPAddr = ""
    pin = ""
    bIsConnected = False
    mac = ""
    bIsActionDone = False
    bIsTVOpen = False

    def __init__(self, device_id='my_device_id', device_name='my_device_name', IPAddr="10.0.0.213", pin='6438', mac="d4:6a:6a:e3:66:af"):
        from bravia_tv import BraviaRC
        # Save input state
        self.device_id = device_id
        self.device_name = device_name
        self.IPAddr = IPAddr
        self.pin = pin
        self.mac = mac

        # Create the BraviaRC agent
        self.agent = BraviaRC(IPAddr, mac=mac)
        self.agent.connect(self.pin, self.device_id, self.device_name)

        self.bIsConnected = self.agent.is_connected()
        self.bIsTVOpen = False
        self.bIsActionDone = False

    def isTVOpen(self):
        return self.bIsTVOpen

    # Return the power status : "off", "active", "standby"
    def GetPowerStatus(self):
        if self.bIsConnected:
            return self.agent.get_power_status()
        return "Unknown"

    def GetVolume(self):
        if self.bIsConnected:
            return self.agent.get_volume_info().get('volume')
        return -1

    def GetPlayingInfo(self):
        if self.bIsConnected:
            return self.agent.get_playing_info()
        return {}

    def GetPlayingChannel(self):
        if self.bIsConnected:
            return self.GetPlayingInfo().get('title')
        return "Unknown"

    #def PlayContent(self):
        # change channel
        #braviarc.play_content(uri)

    # get app list
    #app_info = braviarc.load_app_list()
    #print(app_info)

    # start a given app
    #braviarc.start_app("Netflix")

    # turn on the TV
    def TurnOnTV(self):
        if self.bIsConnected:
            self.agent.turn_on()
            if self.GetPowerStatus() == "active":
                self.bIsTVOpen = True
            else:
                self.bIsTVOpen = False

    # turn off the TV
    def TurnOffTV(self):
        if self.bIsConnected:
            self.agent.turn_off()
            if self.GetPowerStatus() == "active":
                self.bIsTVOpen = True
            else:
                self.bIsTVOpen = False

    def ChangeChannel(self):
        pass

    def DoAction(self,action):
        if self.bIsConnected:
            if action == "Open":
                self.TurnOnTV()
                self.bIsActionDone = True
            elif self.isTVOpen():
                if action == "Close":
                    self.TurnOffTV()
                    self.bIsActionDone = True
                elif action == "Swipe":
                    self.ChangeChannel()
                    self.bIsActionDone = True
            else:
                self.bIsActionDone = False

        return self.bIsActionDone
