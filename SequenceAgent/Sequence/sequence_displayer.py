# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    04/01/2021
# Purpose: Implements "Sequence" for the COMP90054 competitive game environment

# IMPORTS ------------------------------------------------------------------------------------------------------------#

from template import Displayer
from Sequence.sequence_utils import *
import tkinter
from tkinter import font
import copy
import time

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

def make_label(master, x, y, h, w, *args, **kwargs):
    f = tkinter.Frame(master, height=h, width=w)
    f.pack_propagate(0)
    f.place(x=x, y=y)
    label = tkinter.Label(f, *args, **kwargs)
    label.pack(fill=tkinter.BOTH, expand=1)
    return label

class AgentArea():
    def __init__(self, root, agent_id, agent_title, hand_pos, titl_pos, disc_pos):
        self.root = root
        self.agent_id = agent_id
        self.discard = None
        self.discard_pos = disc_pos
        
        #Create agent title labels. Choose font size to best accommodate agent title length.
        text  = "Agent #{}: {}".format(agent_id, agent_title) if len(agent_title)<=20 else agent_title
        fsize = int(15*s) if len(agent_title)<=30 else int(10*s)
        self.agent_title = make_label(root, text=text, x=titl_pos[0], y=titl_pos[1], h=35*s, w=337*s, font=('TkFixedFont', fsize), bg='black', fg='white')
        
        #Each card is offset a certain number of pixels to the right to ensure they are identifiable.
        self.cards = [None]*6
        self.card_pos = [(hand_pos[0]+i*CARD_SEP, hand_pos[1]+20*s) for i in range(6)]
            
    #Place images of cards in this agent's area.
    def update(self, agent, resources):
        for i in range(6):
            try:
                new_image = resources[agent.hand[i]]
                card = self.root.create_image(self.card_pos[i][0], self.card_pos[i][1], image=new_image, tags='card')
                self.cards[i] = card
            except IndexError: #Agent's hand doesn't contain all 6 cards
                self.cards[i] = None #Therefore, don't create new card image, and set card as None
            
        if agent.discard:
            self.root.create_image(self.discard_pos[0], self.discard_pos[1], image=resources[agent.discard], tags='card')
   
     
class BoardArea():
    def __init__(self, root):
        self.root = root
        self.chips = [[None for _ in range(10)] for _ in range(10)]
        self.chip_pos = [[(CHIP_POS[0]+c*CHIP_SEP, CHIP_POS[1]+r*CHIP_SEP) for c in range(10)] for r in range(10)]
    
    #Replace images of chips and drafted cards on the gameboard.
    def update(self, board, resources):
        #Chips.
        for r in range(10):
            for c in range(10):
                chip = board.chips[r][c]
                if not (chip==EMPTY or chip==JOKER):
                    x,y = self.chip_pos[r][c]
                    self.chips[r][c] = self.root.create_image(x, y, image=resources[chip], tags='chip')
        #Drafted cards. Updated after chips in order to stay in the foreground.
        for d in range(len(board.draft)):
            new_image = resources[board.draft[d]]
            self.draft = self.root.create_image(DRFT_POS[0]+d*DRFT_SEP, DRFT_POS[1], image=new_image, tags='card')

                
class GUIDisplayer(Displayer):
    def __init__(self, scale, delay = 0.1):
        self.delay = delay
        # Absolute positions for resources (hands, discards, chips).
        global s,HAND_POS,DISC_POS,TITL_POS,CHIP_POS,DRFT_POS,CHIP_SEP,CARD_SEP,DRFT_SEP,C_WIDTH,C_HEIGHT
        s = 0.5
        HAND_POS = [(124*s,930*s), (124*s,112*s), (1612*s,112*s), (1612*s,930*s)]
        DISC_POS = [(216*s,656*s), (216*s,425*s), (1706*s,425*s), (1706*s,656*s)]
        TITL_POS = [( 48*s,792*s), ( 48*s,257*s), (1537*s,257*s), (1537*s,792*s)]
        CHIP_POS = [536*s,118*s]
        DRFT_POS = [639*s,1080*s]
        CHIP_SEP =   94*s #Separation between board chips.
        CARD_SEP =   37*s #Separation between cards.
        DRFT_SEP =  160*s #Separation between drafted cards.
        C_WIDTH  = 1920*s #Canvas dimensions.
        C_HEIGHT = 1080*s
                
    def InitDisplayer(self, runner):
        #Initialise root frame.
        self.root = tkinter.Tk()
        self.root.title("Sequence! ------ COMP90054 AI Planning for Autonomy")
        self.root.tk.call('wm', 'iconphoto', self.root._w, tkinter.PhotoImage(file='Sequence/resources/icon_main.png'))
        self.root.geometry("{}x{}".format(int(C_WIDTH), int(C_HEIGHT)))
        self.maximised = True
        if s==1: #Fullscreen mode only if running full resolution.
            self.root.attributes("-fullscreen", self.maximised)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)
        
        #Load resources (i.e. images of tabletop, cards, and player chips).
        self.resources = {'table':tkinter.PhotoImage(file="Sequence/resources/background.png").subsample(int(1/s))}
        for rank in ['2','3','4','5','6','7','8','9','t','j','q','k','a']:
            for suit in ['d','c','h','s']:
                card = rank+suit
                self.resources[card] = tkinter.PhotoImage(file="Sequence/resources/cards/{}.png".format(card))
                self.resources[card]  = self.resources[card].subsample(int(1/s))
        for chip in [RED, BLU, RED_SEQ, BLU_SEQ]:
            self.resources[chip] = tkinter.PhotoImage(file="Sequence/resources/chips/{}.png".format(chip))
            self.resources[chip] = self.resources[chip].subsample(int(1/s))
        
        #Initialise canvas and place background table image.
        self.canvas = tkinter.Canvas(self.root, height=C_HEIGHT, width=C_WIDTH, bg='black')
        self.canvas.pack()
        self.table  = self.canvas.create_image(0, 0, image=self.resources['table'], anchor='nw')

        #Generate 4 agent areas.
        self.agent_areas = []
        for i in range(4): 
            area = AgentArea(self.canvas, i, runner.agents_namelist[i%2], HAND_POS[i], TITL_POS[i], DISC_POS[i])
            self.agent_areas.append(area)
            
        #Generate board area.
        self.board_area = BoardArea(self.canvas)

        #Generate scoreboard in separate window.
        self.sb_window = tkinter.Toplevel(self.root)
        self.sb_window.title("Sequence! ------ Activity Log")
        self.sb_window.tk.call('wm', 'iconphoto', self.sb_window._w, tkinter.PhotoImage(file='Sequence/resources/icon_log.png'))
        self.sb_window.geometry("640x455")
        self.sb_frame = tkinter.Frame(self.sb_window)
        self.sb_frame.pack()
        self.scrollbar = tkinter.Scrollbar(self.sb_frame, orient=tkinter.VERTICAL)
        self.move_box=tkinter.Listbox(self.sb_frame,name="actions:", height=37, width=88, selectmode="single", borderwidth=4, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.move_box.yview,troughcolor="white",bg="white")
        self.scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.move_box.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)   
        self.game_state_history=[]
        self.round_num = 0
        self.sb_window.attributes("-topmost", True)
        
    def toggle_fullscreen(self, event=None):
        self.maximised = not self.maximised
        self.root.attributes("-fullscreen", self.maximised)

    def end_fullscreen(self, event=None):
        self.maximised = False
        self.root.attributes("-fullscreen", False)
        
    def _InsertState(self, text, game_state):
        text = text.replace("\n ","")
        self.game_state_history.append(copy.deepcopy(game_state))
        self.move_box.insert(tkinter.END,text)
        self.move_box.see(tkinter.END)
        self.move_box.selection_clear(0, last=None) 
    
    #Rebuild canvas images.
    def _DisplayState(self, game_state):
        #Destroy select canvas images.
        self.canvas.delete('card')
        self.canvas.delete('chip')
        
        #Update displayed areas (agents' hands and discards).
        for agent,area in zip(game_state.agents,self.agent_areas):
            area.update(agent, self.resources)
        #Update displayed board chips.
        self.board_area.update(game_state.board, self.resources)
        self.canvas.update()

    def ExcuteAction(self,player_id,move, game_state):
        self._InsertState(ActionToString(player_id, move, game_state.board.new_seq), game_state)
        self._DisplayState(game_state)
        time.sleep(self.delay)

    def TimeOutWarning(self,runner,id):
        self._InsertState("Agent {} time out, {} out of {}. Choosing random action instead.".format(id, runner.warnings[id],runner.warning_limit),runner.game_rule.current_game_state)
        if id == 0:
            self.move_box.itemconfig(tkinter.END, {'bg':'red','fg':'blue'})
        else:
            self.move_box.itemconfig(tkinter.END, {'bg':'blue','fg':'yellow'})
        pass
        
    def EndGame(self,game_state,scores):
        self._InsertState("--------------End of game-------------",game_state)
        for i,plr_state in enumerate(game_state.agents):
            self._InsertState("Final score with bonus for Agent {}: {}".format(i,plr_state.score),game_state)
        
        self.focus = None
        def OnHistorySelect(event):
            w = event.widget
            self.focus = int(w.curselection()[0])
            if self.focus < len(self.game_state_history):
                self._DisplayState(self.game_state_history[self.focus])
        def OnHistoryAction(event):
            if event.keysym == "Up":
                if self.focus>0:
                    self.move_box.select_clear(self.focus)
                    self.focus -=1
                    self.move_box.select_set(self.focus)
                    if self.focus < len(self.game_state_history):
                        self._DisplayState(self.game_state_history[self.focus])
            if event.keysym == "Down":
                if self.focus<len(self.game_state_history)-1:
                    self.move_box.select_clear(self.focus)
                    self.focus +=1
                    self.move_box.select_set(self.focus)
                    self._DisplayState(self.game_state_history[self.focus])

        self.move_box.bind('<<ListboxSelect>>', OnHistorySelect)
        self.move_box.bind('<Up>', OnHistoryAction)
        self.move_box.bind('<Down>', OnHistoryAction)
    
        self.root.mainloop()
        pass    


class TextDisplayer(Displayer):
    def __init__(self):
        print ("--------------------------------------------------------------------")
        return

    def InitDisplayer(self,runner):
        pass

    def StartRound(self,game_state):
        pass    

    def ExcuteAction(self,i,move, game_state):
        plr_state = game_state.agents[i]
        print("\nAgent {} has chosen the following move:".format(i))
        print(ActionToString(i, move, game_state.board.new_seq))
        print("\n")
        
        print("The new agent state is:")
        print(AgentToString(i, plr_state))
        print ("--------------------------------------------------------------------")
        
    def TimeOutWarning(self,runner,id):
        print ( "Agent {} Time Out, {} out of {}.".format(id,runner.warnings[id],runner.warning_limit))
    
    def EndRound(self,state):
        print("ROUND HAS ENDED")
        print ("--------------------------------------------------------------------")

    def EndGame(self,game_state,scores):
        print("GAME HAS ENDED")
        print ("--------------------------------------------------------------------")
        for plr_state in game_state.agents:
            print ("Score for Agent {}: {}".format(plr_state.id,plr_state.score))

# END FILE -----------------------------------------------------------------------------------------------------------#