//////////////////////////////////////////////
//////////////////////////////////////////////
#define _DO_SERIAL_NOPE

#define PUMP_PIN 3
#define VALVE_PIN 11
#define BUZZER_PIN 12
#define PUMP_INPUT_PIN A3
#define BUZZER_INPUT_PIN A2

#define PUMP 0
#define BUZZER 1


/////////////////////////
#define AUDIO_NOISE_THRESH 20
#define JUICE_THRESH_VAL 200
#define BUZZER_THRESH_VAL 50
#define BUZZER_TOP_VAL 125
#define VALVE_DELAY_VAL 20
#define WAIT_DELAY_VAL 10
#define SIGNAL_OFF_TIME 50

#define BUZZFREQ 60

//////////////////////////////////////////////
//////////////////////////////////////////////
void setup() {

#ifdef _DO_SERIAL
    Serial.begin(9600);
#endif

    pinMode(PUMP_INPUT_PIN, INPUT);
    pinMode(BUZZER_INPUT_PIN, INPUT);
    pinMode(PUMP_PIN, OUTPUT);
    pinMode(VALVE_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    stopReward();  // make sure both the pump and valve are off at startup
    stopBuzzer();
}


/////////////////////////
void loop() {

    long channelOut = -1;

    channelOut = waitForAudioSignal();

#ifdef _DO_SERIAL
    Serial.println("channel = " + String(channelOut));
#endif


    switch (channelOut) {
        case PUMP:
            doJuice();
            break;
        case BUZZER:
            doBuzzer();
            break;
    }
}


//////////////////////////////////////////////
//////////////////////////////////////////////
long waitForAudioSignal() {
    boolean signalPresent = false;

    // wait for input
    while (!signalPresent) {

        if (abs(analogRead(PUMP_INPUT_PIN)) > AUDIO_NOISE_THRESH) {
            signalPresent = true;
            return(PUMP);
        }
        else if (abs(analogRead(BUZZER_INPUT_PIN)) > AUDIO_NOISE_THRESH) {
            signalPresent = true;
            return(BUZZER);
        }
    }
}


//////////////////////////////////////////////
//////////////////////////////////////////////
void doJuice() {

#ifdef _DO_SERIAL
    Serial.println("Juice");
#endif

    startReward();
    waitForAudioOff(PUMP);
    stopReward();

#ifdef _DO_SERIAL
    Serial.println("Juice done");
#endif
}


//////////////////////////////////////////////
//////////////////////////////////////////////
void doBuzzer() {

#ifdef _DO_SERIAL
    Serial.println("Buzz");
#endif

    startBuzzer();
    waitForAudioOff(BUZZER);
    stopBuzzer();

#ifdef _DO_SERIAL
    Serial.println("Buzzer done");
#endif
}


//////////////////////////////////////////////
//////////////////////////////////////////////
void waitForAudioOff(int chan) {
    long startTime = 0;
    long quietTime = 0;
    int ival = 0;

    int inputPin = -1;

    switch(chan) {
        case PUMP:
            inputPin = PUMP_INPUT_PIN;
            break;
        case BUZZER:
            inputPin = BUZZER_INPUT_PIN;
            break;
    }

    startTime = millis();
    quietTime = 0;

    while (quietTime < SIGNAL_OFF_TIME) {
        ival = abs(analogRead(inputPin));

        if (ival > AUDIO_NOISE_THRESH) {
            startTime = millis();
        }

        quietTime = millis() - startTime;
    }
}


//////////////////////////////////////////////
//////////////////////////////////////////////
void startReward() {
    openValve();
    delay(VALVE_DELAY_VAL);
    startPump();
}


//////////////////////////////////////////////
void stopReward() {
    stopPump();
    delay(VALVE_DELAY_VAL);
    closeValve();
}


//////////////////////////////////////////////
void startBuzzer() {
    tone(BUZZER_PIN, BUZZFREQ);
}


//////////////////////////////////////////////
void stopBuzzer() {
    noTone(BUZZER_PIN);
}


//////////////////////////////////////////////
void openValve() {
    digitalWrite(VALVE_PIN, HIGH);
}


//////////////////////////////////////////////
void closeValve() {
    digitalWrite(VALVE_PIN, LOW);
}


//////////////////////////////////////////////
void startPump() {
    digitalWrite(PUMP_PIN, HIGH);
}


//////////////////////////////////////////////
void stopPump() {
    digitalWrite(PUMP_PIN, LOW);
}
