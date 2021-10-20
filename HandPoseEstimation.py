#define all pose estimation
import math
class PoseEstiamtion:
    swipe_lenght = -1
    state = ""
    swipe_tick = 0
    max_tick_swipe = 50 # avg 10 sec since 30 FPS with camera
    Swipe_trigger = True
    first_pos = (-1,-1)
    DontSwipe = False
    prev_state = ""
    action = ""

    def __init__(self):
        self.swipe_lenght = 150
        self.state = "None"
        self.swipe_tick = 0
        self.Swipe_trigger = True
        self.first_pos = (-1,-1)
        self.prev_state = "None"
        self.action = "None"

    def get_Euclidean_Distance(self,a_x,a_y,b_x,b_y):
        return math.sqrt((a_x-b_x)**2 + (a_y-b_y)**2)

    def SwipeTick(self):
        if self.state == "One":
            self.swipe_tick = self.swipe_tick + 1

            if self.swipe_tick > self.max_tick_swipe:
                self.swipe_tick = 0
                self.Swipe_trigger = True
                print("TICK RESET")
    #Each pose function
    # LandMark tuple is (ID, X, Y)
    #   T   F   S   T   F
    #   4   8   12  16  20
    #   3   7   11  15  19
    #   2   6   10  14  18
    #   1   5   9   13  17
    #         0
    # Thumb is up
    def isThumbOpen(self, LandMark):
        if LandMark[3][1] < LandMark[2][1] and LandMark[4][1] < LandMark[2][1]:
            return True
        return False


    def isFirstFingerOpen(self,LandMark):
        if LandMark[7][2] < LandMark[6][2] and LandMark[8][2] < LandMark[6][2]:
            return True
        return False


    def isSecondFingerOpen(self,LandMark):
        if LandMark[11][2] < LandMark[10][2] and LandMark[12][2] < LandMark[10][2]:
            return True
        return False


    def isThirdFingerOpen(self,LandMark):
        if LandMark[15][2] < LandMark[14][2] and LandMark[16][2] < LandMark[14][2]:
            return True
        return False


    def isFourthFingerOpen(self,LandMark):
        if LandMark[19][2] < LandMark[18][2] and LandMark[20][2] < LandMark[18][2]:
            return True
        return False

    def GetAction(self, Landmark):
        if self.ActionSwipeRight(Landmark):
            self.action = "SwipeRight"
        elif self.ActionClose():
            self.action = "Close"
        elif self.ActionOpen():
            self.action = "Open"
        else:
            self.action = "None"
        return self.action

    def GetPose(self, LandMark):
    # TODO: Pass through all the pose and return the estimated pose
        ThumbIsOpen =self.isThumbOpen(LandMark)
        FirstFingerIsOpen = self.isFirstFingerOpen(LandMark)
        SecondFingerIsOpen = self.isSecondFingerOpen(LandMark)
        ThirdFingerIsOpen = self.isThirdFingerOpen(LandMark)
        FourthFingerIsOpen = self.isFourthFingerOpen(LandMark)
        self.prev_state = self.state
        if ThumbIsOpen and FirstFingerIsOpen and SecondFingerIsOpen and ThirdFingerIsOpen and FourthFingerIsOpen:
            #print("FIVE")
            self.state = "Five"
        elif not ThumbIsOpen and FirstFingerIsOpen and SecondFingerIsOpen and ThirdFingerIsOpen and FourthFingerIsOpen:
            #print("FOUR")
            self.state = "Four"
        elif not ThumbIsOpen and FirstFingerIsOpen and SecondFingerIsOpen and ThirdFingerIsOpen and not FourthFingerIsOpen:
            #print("THREE")
            self.state = "Three"
        elif not ThumbIsOpen and FirstFingerIsOpen and SecondFingerIsOpen and not ThirdFingerIsOpen and not FourthFingerIsOpen:
            #print("TWO")
            self.state = "Two"
        elif not ThumbIsOpen and FirstFingerIsOpen and not SecondFingerIsOpen and not ThirdFingerIsOpen and not FourthFingerIsOpen:
            #print("ONE")
            self.state = "One"
        elif not ThumbIsOpen and not FirstFingerIsOpen and not SecondFingerIsOpen and not ThirdFingerIsOpen and not FourthFingerIsOpen:
            #print("FIST")
            self.state = "Fist"

    def ActionSwipeRight(self, LandMark):
        # For swipping, we want the first finger up, while other are down (state ONE)

        if self.state == "One":
            self.SwipeTick()
            #we can assume the hand position will stay the same here therefore we can verify that only 1 keypoint move
            #on a left-to-right axis (x)
            if self.Swipe_trigger:
                self.first_pos = (LandMark[8][1], LandMark[8][2])
                self.Swipe_trigger = False

            #print(self.first_pos[0], self.first_pos[1], LandMark[8][1], LandMark[8][2])
            #print(self.get_Euclidean_Distance(self.first_pos[0],  self.first_pos[1], LandMark[8][1], LandMark[8][2]))
            if self.swipe_lenght <= self.get_Euclidean_Distance(self.first_pos[0], self.first_pos[1], LandMark[8][1], LandMark[8][2]) and not self.Swipe_trigger:
                self.Swipe_trigger = True

                return True
            return False
        else:
            self.Swipe_trigger = True
            return False

    def ActionClose(self):
        if self.state == "Fist" and self.prev_state == "Five":
            return True
        return False

    def ActionOpen(self):
        if self.state == "Five" and self.prev_state == "Fist":
            return True
        return False