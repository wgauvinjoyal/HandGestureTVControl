
#Class to control a tv
class tvcontrol:
    device_id = ""
    device_name = ""
    IPAddr = ""
    pin = ""
    def __init__(self, device_id = 'my_device_id', device_name = 'my_device_name', IPAddr = "10.0.0.213", pin = '6438'):
        from bravia_tv import BraviaRC
        #Save input state
        self.device_id = device_id
        self.device_name = device_name
        self.IPAddr = IPAddr
        self.pin = pin

        #Create the BraviaRC agent
        self.agent = BraviaRC(IPAddr)
        pin = '6438'
        braviarc.connect(pin, device_id, 'my_device_name')


if braviarc.is_connected():
    # get power status
    power_status = braviarc.get_power_status()
    print(power_status)

    # get playing info
    #playing_content = braviarc.get_playing_info()

    # print current playing channel
    #print(playing_content.get('title'))

    # get volume info
    volume_info = braviarc.get_volume_info()

    # print current volume
    print(volume_info.get('volume'))

    # change channel
    #braviarc.play_content(uri)

    # get app list
    #app_info = braviarc.load_app_list()
    #print(app_info)

    # start a given app
    #braviarc.start_app("Netflix")

    # turn off the TV
    braviarc.turn_off()

else:
    print("Failed")