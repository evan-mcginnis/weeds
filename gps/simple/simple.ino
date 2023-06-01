#include <Wire.h>
#include <SFE_MicroOLED.h> 
#include "SparkFun_Ublox_Arduino_Library.h" //http://librarymanager/All#SparkFun_Ublox_GPS

#define PIN_RESET 9
#define DC_JUMPER 1

SFE_UBLOX_GPS myGPS;
MicroOLED oled(PIN_RESET, DC_JUMPER);

const int buttonPin = 2;
const int ledPin =  13;

int buttonState = 0;

void setup() {
  Wire.begin();

  delay(100);
  oled.begin();
  oled.clear(ALL);
  oled.display();
  delay(1000);
  oled.clear(PAGE);
  oled.display();

  oled.setFontType(0);
  oled.setCursor(0,0);

  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
}

void loop() { 
    buttonState = digitalRead(buttonPin);
    if(myGPS.begin() == true){
        myGPS.setI2COutput(COM_TYPE_UBX);
        myGPS.saveConfiguration();

        float latitude = myGPS.getLatitude();
        latitude = latitude / 10000000;

        float longitude = myGPS.getLongitude();
        longitude = longitude / 10000000;
        oled.clear(PAGE);
        oled.setCursor(0,0);
        oled.print("Lat:");
        oled.print(latitude,6);
        oled.print("\nLong:");
        oled.print(longitude,6);
        oled.display();
        delay(10000);
        oled.clear(PAGE);
        oled.display();
        oled.setCursor(0,0);
        oled.print("Acquiring");
        oled.display();
        delay(2000);
    }
    if (buttonState == LOW){
      if (myGPS.begin() == false){
        oled.clear(PAGE);
        oled.print("No GPS");
        oled.display();
        delay(1500);
        oled.clear(PAGE);
        oled.display();
        oled.setCursor(0,0);
      }
      if(myGPS.begin() == true){
        myGPS.setI2COutput(COM_TYPE_UBX);
        myGPS.saveConfiguration();

        float latitude = myGPS.getLatitude();
        latitude = latitude / 10000000;

        float longitude = myGPS.getLongitude();
        longitude = longitude / 10000000;
      ////////////////////////////////////////////
      // Uncomment for altitude, add to output  //
      ////////////////////////////////////////////
      //   float altitude = myGPS.getAltitude();//
      ////////////////////////////////////////////

        oled.clear(PAGE);
        oled.print("Lat:");
        oled.print(latitude,6);
        oled.print("\nLong:");
        oled.print(longitude,6);
        oled.display();
        delay(10000);
        oled.clear(PAGE);
        oled.display();
        oled.setCursor(0,0);
      }
    }

}
