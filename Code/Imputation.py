# -*- coding: utf-8 -*-
import numpy  as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import pickle
os.chdir("./")

def getFOLD_ELEMENT_PARAMETER(RUN_COMMAND_LINE):
    # - READ COMMAND LINE PARAMETERS
    if RUN_COMMAND_LINE==True:
        FoldElement=int(sys.argv[1])
    else:
        FoldElement=0 
    return FoldElement
def getMODE_PARAMETER(RUN_COMMAND_LINE):
    # - READ COMMAND LINE PARAMETERS MODE : "TRAIN" "TEST" "PREPROC"
    if RUN_COMMAND_LINE==True:
        MODE=(sys.argv[2])
    else:
        MODE="IMPUTATION" #"PREPROC"#"TRAINING"#"IMPUTATION"
    return MODE
def print_GLOBAL_PARAMETERS():
    print("----- PARAMETERS -----")
    print("N_FOLDERS:",N_FOLDERS )
    print ("FOLD_ELEMENT:",FOLD_ELEMENT)
    print ("MODE:",MODE)    
RUN_COMMAND_LINE = True # if you are running by command line put the fold of the k-fold number
FOLD_ELEMENT=getFOLD_ELEMENT_PARAMETER(RUN_COMMAND_LINE)
MODE=getMODE_PARAMETER(RUN_COMMAND_LINE)

# PARAMETERS SETTINGS #
####### GLOBAL PARAMETERS ##################
N_FOLDERS=5 # Number of Follds in KFold Validation
FOLD_FIELD_NAME="FOLD"


ID_FIELDNAME="CODICE_INDIVIDUO"

#SELECTED VARIABLE INTO DATASET

INPUT_VARIABLES=["SESSO",
                        "COD_PROV_RESIDENZA",
                        "FL_ITA",
                        "POP_ABC_2017",
                        "TS_APR4_2017",
                        "G_ISTR_2016_CDIFF_CORR",
                        "G_ISTR_2017_CDIFF_CORR",
                        "VAR18",
                        "CLETA_18",                        						
                        "SIREA"]

						
WEIGHT_FIELD_NAME=None #"PESO_CAL_PROV"  # Name of the Sample Weigth Field if you want consider it during the training and imputation phase

TARGET_VARIABLE="TS_MS18_CDIFF" # Name of the Target Variable 

VALUE_MISSING_4_TARGET=[] # None if the target has not missing modality else ['']

						

############################# PREPROCESSING PARAMETERS ###############
SEP="\t"
DatasetFileName="DATASET_WITH_FOLDS.txt"    # Dataset filename in Data directoty
                                            # is needed a FOLD field for the KFold Validation
######################################################################

############################# TRAINING PARAMETERS ###############
TEST_TRIVIAL_PREDICTION=False       # MUST BE False. If it is True put the target variable into input variable 
BS=1024           # BATCH SIZE OF THE TRAINING
NUM_EPOCHS=150   # NUMBER OF EPOCHS
ValSplit=0.1     # VALIDATION SPLITTING OF TRAINING SET
NEURONS_LAYER_ONE=64
NEURONS_LAYER_TWO=32
NEURONS_LAYER_THREE=8
LR=0.05
######################################################################

############################# Imputation PARAMETERS ###############
LoadModelFile = None    # MLP model name in Training Data Directory for imputation phase
                        # if is None the model with last epoch during
						# the trainig is choosen
BS_eval=1024
######################################################################

print_GLOBAL_PARAMETERS()
#------------------------------

#%% 
def RunSelectedMode(MODE,LoadModelFile):
    VARIABILI_SELEZIONATE=INPUT_VARIABLES+["TS_MS18_CDIFF"]
    DATA_DIR=os.sep.join(["..","Data"]) 
    ONEHOT_PICKLE_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"OneHotDict.pkl" 
    TRAIN_MINUS_VAL_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"./valMinusTrain.dsv"
    VAL_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"val.dsv"
    TEST_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"test.dsv"
    TRAIN_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"train.dsv"
    LASTMODEL="lastmodel"+str(FOLD_ELEMENT)+".hdf5"
	#LASTMODEL=DATA_DIR+os.sep+"Training"+os.sep+"lastmodel"+str(FOLD_ELEMENT)+".hdf5"
    def Preprocessing():
    
        def CreaMasterSample_ImputationFile(InputFile,ImputationFile,MasterSampleFile,TargetVariable,targetImpCharList=[''],head=1):
            print('CreaMasterSample_ImputationFile')
            print('\tImputationFile:',ImputationFile)
            print('\tMasterSampleFile:',MasterSampleFile)
            print('\tTargetVariable:',TargetVariable)
            print('\ttargetImpCharList:',targetImpCharList)
            print('\thead:',head)
            #### GESTISCO LE INTESTAZIONI PER I FILE
            ff=open(InputFile)
            headrow=ff.readline().strip() # riga con i nomi dei campi
             
            ImputationFileDescr=open(ImputationFile,'w')
            ImputationFileDescr.write(headrow+"\n")
            MasterSampleFileDescr=open(MasterSampleFile,'w')
            MasterSampleFileDescr.write(headrow+"\n")
            ######################################
            Nrecords=sum(1 for i in open(InputFile, 'rb'))-head
            print ("\t\tInputFile \tNrecords without header: ",Nrecords)
        
            ######################################
            IndexTarget=headrow.split(SEP).index(TargetVariable)
            print("\tTarget is the column in position:",IndexTarget)
            print("\twriting Master Sample and ImputationFile...")
            
            for nn,row in tqdm(enumerate(ff), total=Nrecords):
                target= (row.split(SEP)[IndexTarget])
                if target in targetImpCharList:
                    ImputationFileDescr.write(row)
                else:
                    MasterSampleFileDescr.write(row)
            print("\tend writing")
            ff.close()
            MasterSampleFileDescr.close()
            ImputationFileDescr.close()
            Nrecords=sum(1 for i in open(MasterSampleFile, 'rb'))-head
            print ("\t\tMasterSampleFile \tNrecords without header: ",Nrecords)
            Nrecords=sum(1 for i in open(ImputationFile, 'rb'))-head
            print ("\t\tImputationFile \tNrecords without header: ",Nrecords)

        def CreaDataSetTraiTestKFOLD(InputFileMaster,TestFile,TrainFile,FOLD_ELEMENT,head=1):

            def index_of_train_testFOLD():
                FOLD_Test=[FOLD_ELEMENT]
                FOLD_Train=list(range(N_FOLDERS))
                FOLD_Train.remove(FOLD_ELEMENT)
                            
                print('\tFOLD_Train:',FOLD_Train)
                print('\tFOLD_Test:',FOLD_Test)
                return FOLD_Train,FOLD_Test
            
            
            def individuaIlCampoFOLD(header,field=FOLD_FIELD_NAME):
                print(header)
                print("sep",SEP,header.split(SEP))
                indexField=header.split(SEP).index(field)
                return indexField
            
            print('\n\n')
            print('-------------CreaDataSetTraiTestKFOLD')
            print('\tInputFileMaster:',InputFileMaster)
            print('\tTestFile:',TestFile)
            print('\tTrainFile:',TrainFile)
            print('\head:',head)
            
            FOLD_Train,FOLD_Test=index_of_train_testFOLD()
            
            # BUILD DATASET DI TRAINING - TEST
        
            ####### LIST FOR  TRAINING E TESTING
            
            
            #### 
            ff=open(InputFileMaster)
            header=ff.readline()
            header=header.strip()
            
            
            Nrecord=sum(1 for i in open(InputFileMaster, 'rb'))-int(head)
            FOLD_Field_Index=individuaIlCampoFOLD(header,field=FOLD_FIELD_NAME)
            print("indice campo FOLD:",FOLD_Field_Index)
            
            
            TestFileDescr=open(TestFile,'w')
            TestFileDescr.write(header+"\n")
            TrainFileDescr=open(TrainFile,'w')
            TrainFileDescr.write(header+"\n")
            ######################################
            
            
            ########## WRITING RECORDS
            
            print("---------writing  Dataset...")
            
            for nn,row in tqdm(enumerate(ff), total=Nrecord):
                FOLD_row_i=float(row.replace("\n","").split(SEP)[FOLD_Field_Index])
                
                if FOLD_row_i in FOLD_Train:
                    TrainFileDescr.write(row)
                if FOLD_row_i in FOLD_Test:
                    TestFileDescr.write(row)
                
            print("----------end writing")
            
            TrainFileDescr.close()
            TestFileDescr.close()
            
            Nrecords=sum(1 for i in open(InputFileMaster, 'rb'))-int(head)
            print ("\t",InputFileMaster,"\tNrecords without header: ",Nrecords)
        
            Nrecords=sum(1 for i in open(TestFile, 'rb'))-head
            print ("\t",TestFile,"\tNrecords without header: ",Nrecords)
            
            Nrecords=sum(1 for i in open(TrainFile, 'rb'))-head
            print ("\t",TrainFile,"\tNrecords without header: ",Nrecords)
        #################################CreaDataSetTraiTestKFOLD  
            
            
        def print_PREPROC_PARAMETERS():        
            print("----- PARAMETERS PREPROC-----")
            print ("\tINPUT_FILE:",INPUT_FILE)
            print ("\tMASTER_SAMPLE_FILE:",MASTER_SAMPLE_FILE)
            print ("\tIMPUTATION_FILE:",IMPUTATION_FILE)
            print ("\tSEP:",SEP)
            print ("\tTEST_FILE:",TEST_FILE)
            print ("\tTRAIN_FILE:",TRAIN_FILE)
            
        def WriteEncodingOneHot():
            
            FILE_LETTURA_MODALITA=MASTER_SAMPLE_FILE

            def CreateIndexForSelectedVariable(HeaderFile,variabili_selezionate):
                print("\n\n\n\tCreateIndexForSelectedVariable:")
                print("\tHeaderFile:",HeaderFile)
                print("\tvariabili_selezionate:",variabili_selezionate)
                print("\tsep:",SEP)
                dfInput=pd.read_csv(HeaderFile,sep=SEP,nrows=10)
                variabili=list(dfInput.columns)
                index_variabili_selezionate = [variabili.index(rom)for rom in variabili_selezionate]
                del dfInput
                return sorted(index_variabili_selezionate),variabili
                 
            def CreateOneHotEncoder(index_variabili_selezionate,variabili,InputFile):    
                print('\n\n\n')
                print('CreateOneHotEncoder')
                print('\tIndex_variabili_selezionate',index_variabili_selezionate)
                print('\tVariabili',variabili)
                
                def print_OneHotEncoder(OneHotEncoder):
                    for key in OneHotEncoder:
                        print(key)
                        for modality in OneHotEncoder[key]:
                            print (modality,OneHotEncoder[key][modality])            
                
                def fromList2OneHot(listaModalita):
                    listaModalita.sort()
                    listaModalita
                    dummy={}
                    for modalita in listaModalita:
                        OneHotArray=np.zeros((len(listaModalita)),dtype=np.int8) 
                        OneHotArray[listaModalita.index(modalita)]=1
                        dummy[str(modalita)]=OneHotArray
                    return dummy

                OneHotEncoder={}
                for field in index_variabili_selezionate:
                    if field == index_TargetVariable:
                        missing= VALUE_MISSING_4_TARGET
                    else:
                        missing=['']
                
                    print("\t",field,variabili[field])
                    
                    dfField=pd.read_csv(InputFile,sep=SEP,usecols=[field],dtype=str)                
                    print("\t","MODALITY",InputFile," FOR THE VARIABLE",field)

                    modalita=dfField[variabili[field]].value_counts()
                    print("\tMODALITY: ",len(modalita))

                    OneHotEncoder[variabili[field]]=fromList2OneHot(missing+list(modalita.index))
                    print("\n")
                    
                print_OneHotEncoder(OneHotEncoder)   
                
                
                
                return OneHotEncoder
                  
            index_variabili_selezionate,NomeTutteVariabili=CreateIndexForSelectedVariable(MASTER_SAMPLE_FILE,VARIABILI_SELEZIONATE)
            
            index_TargetVariable= NomeTutteVariabili.index(TARGET_VARIABLE)
            print("Target Variable Index:",index_TargetVariable)

            OneHotEncoder=CreateOneHotEncoder(index_variabili_selezionate,NomeTutteVariabili,FILE_LETTURA_MODALITA)
            
            
            with open(ONEHOT_PICKLE_FILE,"wb") as f:
                preprocDump=(OneHotEncoder,NomeTutteVariabili,index_variabili_selezionate,index_TargetVariable)
                pickle.dump(preprocDump, f, pickle.HIGHEST_PROTOCOL)


        print_PREPROC_PARAMETERS()        
        CreaMasterSample_ImputationFile(INPUT_FILE, IMPUTATION_FILE, MASTER_SAMPLE_FILE, TARGET_VARIABLE)
        CreaDataSetTraiTestKFOLD(MASTER_SAMPLE_FILE,TEST_FILE,TRAIN_FILE,FOLD_ELEMENT)        
        WriteEncodingOneHot()        


##################################################################################
    def loadOneHotPickle(ONEHOT_PICKLE_FILE):
        with open(ONEHOT_PICKLE_FILE,"rb") as f:
            preprocDump=pickle.load(f)    
            return preprocDump
           
    class OneHotGenerator():
        def __init__(self,fileInputGenerator,batchSize,OneHotEncoder,nomi_variabili,index_TargetVariable,weight_field_name=None,mode="eval"):
            self.weight_field_name=weight_field_name
            self.OneHotEncoder=OneHotEncoder
            self.fileInputGenerator=fileInputGenerator
            self.batchSize=batchSize
            self.nomi_variabili=nomi_variabili
            self.index_TargetVariable=index_TargetVariable
            self.mode=mode
            self.count=0
            self.tuplalist=[]
        def get_OneHotGenerator(self):
            ##print("START----")
            f =open(self.fileInputGenerator,"r",encoding="utf8")
            header  = f.readline()
            header=header.strip().split(SEP)
            
            if self.weight_field_name is not None:
                self.weightIndex=header.index(self.weight_field_name)
                print(header,self.weightIndex)
            self.count=0
            while True:
                OneHotBatch=[]
                TargetBatch=[]
                weightBatch=[]
                ##print("BATCH RESET---- ",count)
                
                while len(OneHotBatch) < self.batchSize:
                    outOnehot=[]
                    line = f.readline()
                    
                    # check to see if the line is empty, indicating we have
                    # reached the end of the file
                    ##print("CONTROLLO FINE FILE SE LA RIGA SUCCESSIVA E' FINE DEL FILE E ' ENTRATO NEL IF")
                    if line == "":
                        ##print(mode+" FINE DEL FILE: "+fileInputGenerator)
                        # reset the file pointer to the beginning of the file
                        # and re-read the line
                        f.seek(0)
                        self.count=0
                        #f.close()
                        #f =open(fileInputGenerator,"r",encoding="utf8")
                        line = f.readline()
                        line = f.readline()
                        ##print(line)
                        # if we are evaluating we should now break from our
                        # loop to ensure we don't continue to fill up the
                        # batch from samples at the beginning of the file
                        if self.mode == "eval":
                            break
                    line = line.strip().split(SEP)
                    
                    
                    if self.weight_field_name is not None:
                        weightBatch.append(float(line[self.weightIndex].replace(",",".")))
                    
                    #print(variabili[get_Target()],line[get_Target()])
                    
                    #print("OneHotEncoder ",line[get_Target()],OneHotEncoder[variabili[get_Target()]][line[get_Target()]])
                    #print(self.nomi_variabili)
                    TargetBatch.append(self.OneHotEncoder[self.nomi_variabili[self.index_TargetVariable]][line[self.index_TargetVariable]])
                    #TargetBatch.append(np.zeros(10))
                    for i in get_inputVariable:
                        try:
                            outOnehot.extend(self.OneHotEncoder[self.nomi_variabili[i]][line[i]])
                        except:
                            print(self.nomi_variabili[i],line[i]) 
                            exit(0)
                            
                    OneHotBatch.append(outOnehot)
        
                # yield the batch to the calling function
                np_OneHotBatch=np.array(OneHotBatch)
                np_TargetBatch=np.array(TargetBatch)
                
                if self.weight_field_name is not None:
                    np_weightBatch=np.array(weightBatch)
                    batch=(np_OneHotBatch, np_TargetBatch,np_weightBatch)
                else:
                    batch=(np_OneHotBatch, np_TargetBatch)
                    
                self.count=self.count+len(np_OneHotBatch)
                #tupla=("LEN np_OneHotBatch: ",self.fileInputGenerator,len(np_OneHotBatch),self.count)
                #self.tuplalist.append(tupla)
                
                yield batch              
                           
    def TrainingMLP(BS,NUM_EPOCHS,TrainFile=None,ValFile=None):
        
            
            from keras.models import Sequential
            from keras.layers import Dense,Dropout
            from keras.optimizers import SGD
            from keras.callbacks import ModelCheckpoint
 
            def get_data_target_dim():
                
                
                dummy=OneHotGenerator(TrainFile, 16,OneHotEncoder,nomi_variabili,index_TargetVariable,WEIGHT_FIELD_NAME)
                dummyGen = dummy.get_OneHotGenerator()
                input_dim=next(dummyGen)[0].shape[1]
                output_dim=next(dummyGen)[1].shape[1] 
                return input_dim,output_dim


            def Build_MPL_model():
                model=Sequential()
                model.add(Dense (NEURONS_LAYER_ONE,activation='relu',input_dim=input_dim))
                model.add(Dropout(0.5))
                model.add(Dense (NEURONS_LAYER_TWO,activation='relu'))
                model.add(Dropout(0.5))                
                model.add(Dense (NEURONS_LAYER_THREE,activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(output_dim,activation='softmax'))
                model.summary()
                return model


            
            NUM_TEST=sum(1 for i in open(ValFile, 'rb'))-1
            NUM_TRAIN=sum(1 for i in open(TrainFile, 'rb'))-1
            print("BS: ",BS)
            print("NUM_EPOCHS: ",NUM_EPOCHS)
            print("NUM_TEST: ",NUM_TEST)
            print("NUM_TRAIN: ",NUM_TRAIN)
        
                    
            #######################################
            ####### TrainingMLP MAIN ##############
            #######################################        
            
            
            input_dim,output_dim=get_data_target_dim()            

            model=Build_MPL_model()        
            sgd = SGD(lr=LR, decay=1e-7, momentum=0.9, nesterov=True)
            #sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
            
            model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

            trainOneHotGenerator=OneHotGenerator(TrainFile, BS,OneHotEncoder,nomi_variabili,index_TargetVariable,WEIGHT_FIELD_NAME,mode="eval")
            
            valOneHotGenerator=OneHotGenerator(ValFile, BS,OneHotEncoder,nomi_variabili,index_TargetVariable,WEIGHT_FIELD_NAME,mode="eval")
            
            trainGen = trainOneHotGenerator.get_OneHotGenerator()
            
            valGen = valOneHotGenerator.get_OneHotGenerator()

            filepath=DATA_DIR+os.sep+"Training"+os.sep+"weights-improvement-"+str(FOLD_ELEMENT)+"-{epoch:02d}-{val_loss:.2f}.hdf5"

            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            callbacks_list = [checkpoint]

            steps_train=NUM_TRAIN//BS+bool(NUM_TRAIN%BS)

            steps_test=NUM_TEST//BS+bool(NUM_TEST%BS)


            model.fit_generator(trainGen,steps_per_epoch=steps_train,   validation_data=valGen,validation_steps=steps_test,epochs=NUM_EPOCHS, callbacks=callbacks_list,verbose=2)
            
            model.save(DATA_DIR+os.sep+"Training"+os.sep+LASTMODEL)

            print(trainOneHotGenerator.tuplalist)
            print("")
            print(valOneHotGenerator.tuplalist)
    
            return model

    def RunImputation(LoadModelFile):
        def EvaluationMLP(ImputationFile,LoadModelFile,ImputationOutFile):
            dir()
            print("EvaluationMLP")
            print('\tImputationFile: ',ImputationFile)
            print("\tLoadModelFile: ",LoadModelFile)
            #ImputationFile=MasterSampleFile

            from keras.models import load_model
            global model
            if LoadModelFile is not None:         
                model = load_model(DATA_DIR+os.sep+"Training"+os.sep+LoadModelFile,compile=False)
            model.summary()
            #print(model.summary())    
            NUM_IMP=sum(1 for i in open(ImputationFile, 'rb'))-1
            steps=NUM_IMP//BS_eval+bool(NUM_IMP%BS_eval)
            print("\tSteps Evaluation",steps)
            
            
            ImpOneHotGenerator=OneHotGenerator(ImputationFile, BS_eval,OneHotEncoder,nomi_variabili,index_TargetVariable,mode="eval")
            ImpGen = ImpOneHotGenerator.get_OneHotGenerator()

            
            p=model.predict_generator(ImpGen,steps=steps,verbose=0)
            
            #########################
            Pred2Label={}
            for k in OneHotEncoder[TARGET_VARIABLE]:
                v=np.argmax(OneHotEncoder[TARGET_VARIABLE][k],axis=0)
                Pred2Label[v]=k
            def f(x):
                return Pred2Label[x]
            ##########################
            
            df=pd.read_csv(ImputationFile,sep=SEP,usecols=AddVariable4OutputImputationFile)

            def predictionClass(distribution):
                import random
                # returns a modality according to a given probability distribution 
                x=random.random()
                cum_i_prev=0
                for i in range(len(distribution)):
                    cum_i=sum(distribution[:i+1])
                    #print ("intervallo",cum_i_prev,cum_i)
                    if (cum_i_prev<x<cum_i):
                        return i
            
            def predictionClassXIndividuals(distr_x_indiv):
                # returns an array of classes for each individual
                classe_x_individuo=[]
                for distrib in distr_x_indiv:
                     classe_x_individuo.append(predictionClass(distrib))
                return np.array(classe_x_individuo)





            #if (IMPUTATION_MODE=="ModalClass"):                
            PredictionModal=pd.DataFrame(np.argmax(p,axis=1))
            #if (IMPUTATION_MODE=="DistrExtraction"):
            PredictionDistr=pd.DataFrame(predictionClassXIndividuals(p))


            df['ClassDistr']=PredictionDistr
            df['ClassDistr']=df.ClassDistr.apply(f)
            
            df['ClassModal']=PredictionModal
            df['ClassModal']=df.ClassModal.apply(f)            
                        
            

            DFout=pd.DataFrame(p)
            DFout=pd.concat((df,DFout),axis=1)
            DFout.to_csv(ImputationOutFile,sep=";",index=False)
            
            ###################################
            def printPerformance(CLASS_TYPE):
                from scipy.stats import entropy
                print("--------"+CLASS_TYPE+"------------")
                CAT1=pd.Categorical(df[TARGET_VARIABLE],categories=[1,2,3,4,5,6,7,8])
                CAT2=pd.Categorical(df[CLASS_TYPE],categories=['1','2','3','4','5','6','7','8'])
                Table=pd.crosstab(CAT1,CAT2,dropna=False,colnames=["pred"],rownames=["real"])
                print(Table)   
                real=Table.values.sum(axis=1)
                pred=Table.values.sum(axis=0)
                
                print("real",np.abs(real))
                print("pred",np.abs(pred))

                f_real=real/np.sum(real)
                f_pred=pred/np.sum(pred)
                KL=entropy(pred, real,base=2)


                print("diff abs",np.mean(np.abs(f_real-f_pred)))
                print("diff rel",np.mean(np.abs(f_real-f_pred)/f_real)*100)
                print("KL",KL)
                print("ACC:",np.trace(Table.values)/np.sum(Table.values)*100)
                
            printPerformance('ClassDistr')
            printPerformance('ClassModal')
        EvaluationMLP(ImputationFile,LoadModelFile,ImputationOutFile)
    
    def get_inputVariable():
        if (TEST_TRIVIAL_PREDICTION==True):
            get_inputVariable=[i for i in index_variabili_selezionate  ]
            print ("TEST_TRIVIAL_PREDICTION TRUE "*100)
        else:
            get_inputVariable=[i for i in index_variabili_selezionate if i!=index_TargetVariable ]
        return get_inputVariable

    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
             
    if MODE == "PREPROC":
        print("PREPROC MODE RUNNING...")
        # PARAMETERS PREPROC SETTINGS 
        INPUT_FILE=DATA_DIR+os.sep+DatasetFileName
        IMPUTATION_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"ImputationFile.dsv"
        MASTER_SAMPLE_FILE=DATA_DIR+os.sep+"Preproc"+os.sep+"MasterSampleFileFile.dsv"
        Preprocessing()
               
        
    if MODE == "TRAINING":
        print("TRAINING MODE RUNNING...")        
     
        (OneHotEncoder,nomi_variabili,index_variabili_selezionate,index_TargetVariable)=loadOneHotPickle(ONEHOT_PICKLE_FILE)

        get_inputVariable=get_inputVariable()  
        
        head=1
        
        ff=open(TRAIN_FILE)
        header=ff.readline()
        header=header.strip()
        Nrecord=sum(1 for i in open(TRAIN_FILE, 'rb'))-int(head)
        ValFileDescr=open(VAL_FILE,'w')
        ValFileDescr.write(header+"\n")
        TrainMinusValFileDescr=open(TRAIN_MINUS_VAL_FILE,'w')
        TrainMinusValFileDescr.write(header+"\n")
        N_VAL_RECORDS=int(Nrecord*ValSplit)
        val_row_list=np.random.RandomState(seed=42245).permutation(np.array(list(range(Nrecord))))[:N_VAL_RECORDS]
        i=0
        
        for nn,row in tqdm(enumerate(ff), total=Nrecord):
            if (i in val_row_list):
                ValFileDescr.write(row)
            else:
                TrainMinusValFileDescr.write(row)
            i+=1
            
        ValFileDescr.close()
        TrainMinusValFileDescr.close()
        
        model =TrainingMLP(BS,NUM_EPOCHS,TrainFile=TRAIN_MINUS_VAL_FILE,ValFile=VAL_FILE)
        
    
    if MODE == "IMPUTATION":

        AddVariable4OutputImputationFile=[ID_FIELDNAME,TARGET_VARIABLE]
        if WEIGHT_FIELD_NAME is not None:
            AddVariable4OutputImputationFile.append(WEIGHT_FIELD_NAME)

        
        ImputationFile=TEST_FILE
        
        if LoadModelFile is None:
            LoadModelFile=LASTMODEL
        ImputationOutFile=DATA_DIR+os.sep+"Imputation"+os.sep+"outfile"+str(FOLD_ELEMENT)+".out"
        
        (OneHotEncoder,nomi_variabili,index_variabili_selezionate,index_TargetVariable)=loadOneHotPickle(ONEHOT_PICKLE_FILE)
        get_inputVariable=get_inputVariable()  
        RunImputation(LoadModelFile)
     
#### MAIN ############
RunSelectedMode(MODE,LoadModelFile)
