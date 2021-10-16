
#Class to control a tv
class TVControl:
    device_id = ""
    device_name = ""
    IPAddr = ""
    pin = ""
    IsConnected = False

    def __init__(self, device_id='my_device_id', device_name='my_device_name', IPAddr="10.0.0.213", pin='6438'):
        from bravia_tv import BraviaRC
        # Save input state
        self.device_id = device_id
        self.device_name = device_name
        self.IPAddr = IPAddr
        self.pin = pin

        # Create the BraviaRC agent
        self.agent = BraviaRC(IPAddr)
        self.agent.connect(self.pin, self.device_id, self.device_name)
        self.IsConnected = self.agent.is_connected()

    # Return "On" or "Off"
    def GetPowerStatus(self):
        if self.IsConnected:
            return self.agent.get_power_status()
        return "Unknown"

    def GetVolume(self):
        if self.IsConnected:
            return self.agent.get_volume_info().get('volume')
        return -1

    def GetPlayingInfo(self):
        if self.IsConnected:
            return self.agent.get_playing_info()
        return {}

    def GetPlayingChannel(self):
        if self.IsConnected:
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

    # turn off the TV
    def TurnOffTV(self):
        if self.IsConnected:
            self.agent.turn_off()
