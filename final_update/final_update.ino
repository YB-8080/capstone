

#include <WiFi.h>
#include <Wire.h>
#include <Preferences.h>
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"

#include <Adafruit_SHT31.h>
#include <Adafruit_SGP30.h>
#include <math.h>

/* WIFI */
#define WIFI_SSID     "ESP_WiFi"
#define WIFI_PASSWORD "12345678"

/* FIREBASE */
#define API_KEY      "AIzaSyBftCPD1UFads1oELW2bRUOu1_F_Jn0S1w"
#define DATABASE_URL "https://firesens-1a708-default-rtdb.firebaseio.com"
#define FIRE_PATH "/fireSensors/current"

/* I2C */
#define I2C_SDA 21
#define I2C_SCL 22

/* MQ ADC1 pins (WiFi safe) */
static const int PIN_MQ2 = 34; // smoke
static const int PIN_MQ4 = 35; // CH4
static const int PIN_MQ6 = 32; // LPG
static const int PIN_MQ9 = 33; // CO

/* GUVA UV analog pin */
static const int PIN_UV = 39;  // ADC1, WiFi safe

/* SDS011 UART pins */
#define SDS_RX 16   // ESP32 RX2 <- SDS011 TX
#define SDS_TX 17   // ESP32 TX2 -> SDS011 RX

/* Force calibration (BOOT button) */
static const int PIN_FORCE_CAL = 0;

/* ADC */
static const float ADC_VREF = 3.3f;
static const int   ADC_MAX  = 4095;

/* MQ board powered by 5V */
static const float VC_DIVIDER = 5.0f;

/* Voltage divider factor: Vadc = Vrl_actual * DIV_FACTOR */
static const float DIV_FACTOR = 0.6667f; // 

/* RL values (Ohms) */
static const float RL_MQ2 = 1700.0f;
static const float RL_MQ4 = 1800.0f;
static const float RL_MQ6 = 1700.0f;
static const float RL_MQ9 = 1000.0f;

/* Clean-air ratios (approx) */
static const float RATIO_MQ2_CLEAN_AIR = 9.83f;
static const float RATIO_MQ4_CLEAN_AIR = 4.40f;
static const float RATIO_MQ6_CLEAN_AIR = 10.0f;
static const float RATIO_MQ9_CLEAN_AIR = 9.60f;

/* Curves: Rs/R0 = A * (ppm^B)  => ppm = ((Rs/R0)/A)^(1/B) */
static const float MQ2_SMOKE_A = 30.9f;
static const float MQ2_SMOKE_B = -0.428f;

static const float MQ4_CH4_A   = 15.7f;
static const float MQ4_CH4_B   = -0.398f;

static const float MQ6_LPG_A   = 37.1f;
static const float MQ6_LPG_B   = -0.523f;

static const float MQ9_CO_A    = 8.09f;
static const float MQ9_CO_B    = -0.318f;

/* Timing */
const unsigned long SEND_INTERVAL_MS = 5000;
const unsigned long CAL_MS           = 20000;  // calibration window
const unsigned long SGP_INTERVAL_MS  = 1000;   // IAQmeasure every 1s

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

Adafruit_SHT31 sht31;
Adafruit_SGP30 sgp;
bool shtOk = false, sgpOk = false;

Preferences prefs;
HardwareSerial SDS(2);

unsigned long lastSendMs = 0;
unsigned long lastSgpMs  = 0;

/* Saved R0 values */
float r0_mq2 = 0, r0_mq4 = 0, r0_mq6 = 0, r0_mq9 = 0;

/* Latest SDS011 values */
float sds_pm25 = 0.0f;
float sds_pm10 = 0.0f;
bool  sdsOk    = false;

/* ---------- Helpers ---------- */

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\n WiFi Connected");
  Serial.print("IP: "); Serial.println(WiFi.localIP());
}

int readAnalogAvg(int pin, int samples = 20) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  int v = (int)(sum / samples);
  if (v < 0) v = 0;
  if (v > ADC_MAX) v = ADC_MAX;
  return v;
}

/* ESP32 ADC voltage */
float adcToVadc(int adc) {
  return (adc * ADC_VREF) / (float)ADC_MAX;
}

/* actual VRL on MQ board */
float vadcToVrlActual(float vadc) {
  if (DIV_FACTOR <= 0.0001f) return vadc;
  return vadc / DIV_FACTOR;
}

/* Rs = (Vc/VRL - 1) * RL */
float calcRs(float vrl_actual, float vc, float rl) {
  if (vrl_actual <= 0.001f) return 1e9f;
  float rs = (vc / vrl_actual - 1.0f) * rl;
  if (!isfinite(rs) || rs < 0) rs = 1e9f;
  return rs;
}

/* ppm from Rs/R0 curve */
float calcPpmFromCurve(float rs, float ro, float A, float B) {
  if (ro <= 0.0f) return 0.0f;
  float ratio = rs / ro;
  if (ratio <= 0.0f) return 0.0f;
  float ppm = powf(ratio / A, 1.0f / B);
  if (!isfinite(ppm) || ppm < 0) ppm = 0.0f;
  return ppm;
}

/* Calibrate R0 in clean air */
float calibrateRo(int pin, float clean_ratio, float rl) {
  double acc = 0.0;
  int n = 0;
  unsigned long t0 = millis();

  while (millis() - t0 < CAL_MS) {
    int adc = readAnalogAvg(pin, 10);
    float vadc = adcToVadc(adc);
    float vrl  = vadcToVrlActual(vadc);
    float rs   = calcRs(vrl, VC_DIVIDER, rl);
    acc += rs;
    n++;
    delay(80);
    yield();
  }

  if (n <= 0) return 0.0f;
  float rs_air = (float)(acc / (double)n);
  return rs_air / clean_ratio;
}

void saveRo() {
  prefs.begin("mq_ro", false);
  prefs.putBool("has", true);
  prefs.putFloat("ro2", r0_mq2);
  prefs.putFloat("ro4", r0_mq4);
  prefs.putFloat("ro6", r0_mq6);
  prefs.putFloat("ro9", r0_mq9);
  prefs.end();
}

bool loadRo() {
  prefs.begin("mq_ro", true);
  bool has = prefs.getBool("has", false);
  if (has) {
    r0_mq2 = prefs.getFloat("ro2", 0);
    r0_mq4 = prefs.getFloat("ro4", 0);
    r0_mq6 = prefs.getFloat("ro6", 0);
    r0_mq9 = prefs.getFloat("ro9", 0);
  }
  prefs.end();
  return has;
}

void doCalibration() {
  Serial.println("\n MQ R0 calibration started (CLEAN AIR)...");
  Serial.println("   Keep away from smoke/gas for ~20 seconds.");
  delay(1500);

  r0_mq2 = calibrateRo(PIN_MQ2, RATIO_MQ2_CLEAN_AIR, RL_MQ2);
  r0_mq4 = calibrateRo(PIN_MQ4, RATIO_MQ4_CLEAN_AIR, RL_MQ4);
  r0_mq6 = calibrateRo(PIN_MQ6, RATIO_MQ6_CLEAN_AIR, RL_MQ6);
  r0_mq9 = calibrateRo(PIN_MQ9, RATIO_MQ9_CLEAN_AIR, RL_MQ9);

  Serial.printf("R0 saved: MQ2=%.2f MQ4=%.2f MQ6=%.2f MQ9=%.2f\n",
                r0_mq2, r0_mq4, r0_mq6, r0_mq9);

  saveRo();
}

/* ---------- SDS011 ---------- */
/* Packet: AA C0 PM25L PM25H PM10L PM10H ID1 ID2 CHECKSUM AB */
bool readSDS011(float &pm25, float &pm10) {
  while (SDS.available() >= 10) {
    if (SDS.read() == 0xAA) {
      uint8_t buf[9];
      if (SDS.readBytes(buf, 9) == 9) {
        if (buf[0] == 0xC0 && buf[8] == 0xAB) {
          uint8_t sum = 0;
          for (int i = 0; i < 6; i++) sum += buf[i + 1];

          if (sum == buf[7]) {
            uint16_t pm25raw = (uint16_t)buf[1] | ((uint16_t)buf[2] << 8);
            uint16_t pm10raw = (uint16_t)buf[3] | ((uint16_t)buf[4] << 8);

            pm25 = pm25raw / 10.0f;
            pm10 = pm10raw / 10.0f;
            return true;
          }
        }
      }
    }
  }
  return false;
}

/* ---------- GUVA UV ---------- */
float readUVVoltage() {
  int adc = readAnalogAvg(PIN_UV, 20);
  return adcToVadc(adc);
}

/* Simple estimate */
float voltageToUVIndex(float v) {
  float uvi = v * 10.0f;
  if (uvi < 0) uvi = 0;
  return uvi;
}

/* ---------- Setup ---------- */

void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(PIN_FORCE_CAL, INPUT_PULLUP);
  pinMode(PIN_UV, INPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  SDS.begin(9600, SERIAL_8N1, SDS_RX, SDS_TX);

  connectWiFi();

  Wire.begin(I2C_SDA, I2C_SCL);

  // SHT31: try both addresses if needed
  shtOk = (sht31.begin(0x44) || sht31.begin(0x45));
  sgpOk = sgp.begin();

  Serial.println(shtOk ? "SHT3x OK" : " SHT3x missing (will send 0)");
  Serial.println(sgpOk ? " SGP30 OK" : " SGP30 missing (will send 0)");
  Serial.println(" SDS011 initialized");
  Serial.println(" GUVA UV sensor initialized");

  /* Firebase */
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback;

  Firebase.signUp(&config, &auth, "", "");
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  // Load or calibrate R0
  bool hasRo = loadRo();
  if (!hasRo || digitalRead(PIN_FORCE_CAL) == LOW) {
    doCalibration();
  } else {
    Serial.printf("Using saved R0: MQ2=%.2f MQ4=%.2f MQ6=%.2f MQ9=%.2f\n",
                  r0_mq2, r0_mq4, r0_mq6, r0_mq9);
  }

  Serial.println(" Setup Complete\n");
}

/* ---------- Loop ---------- */

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();

  // update SGP30 every 1 second
  if (sgpOk && millis() - lastSgpMs >= SGP_INTERVAL_MS) {
    lastSgpMs = millis();
    sgp.IAQmeasure();
  }

  if (millis() - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = millis();

    /* MQ read -> Rs */
    int a2 = readAnalogAvg(PIN_MQ2);
    int a4 = readAnalogAvg(PIN_MQ4);
    int a6 = readAnalogAvg(PIN_MQ6);
    int a9 = readAnalogAvg(PIN_MQ9);

    float v2 = vadcToVrlActual(adcToVadc(a2));
    float v4 = vadcToVrlActual(adcToVadc(a4));
    float v6 = vadcToVrlActual(adcToVadc(a6));
    float v9 = vadcToVrlActual(adcToVadc(a9));

    float rs2 = calcRs(v2, VC_DIVIDER, RL_MQ2);
    float rs4 = calcRs(v4, VC_DIVIDER, RL_MQ4);
    float rs6 = calcRs(v6, VC_DIVIDER, RL_MQ6);
    float rs9 = calcRs(v9, VC_DIVIDER, RL_MQ9);

    /* ppm (estimated) */
    float ppm2 = calcPpmFromCurve(rs2, r0_mq2, MQ2_SMOKE_A, MQ2_SMOKE_B);
    float ppm4 = calcPpmFromCurve(rs4, r0_mq4, MQ4_CH4_A,   MQ4_CH4_B);
    float ppm6 = calcPpmFromCurve(rs6, r0_mq6, MQ6_LPG_A,   MQ6_LPG_B);
    float ppm9 = calcPpmFromCurve(rs9, r0_mq9, MQ9_CO_A,    MQ9_CO_B);

    /* I2C sensors */
    float t = (shtOk) ? sht31.readTemperature() : 0.0f;
    float h = (shtOk) ? sht31.readHumidity()    : 0.0f;

    uint16_t tvoc = (sgpOk) ? sgp.TVOC : 0;
    uint16_t eco2 = (sgpOk) ? sgp.eCO2 : 0;
    if (tvoc == 0xFFFF) tvoc = 0;
    if (eco2 == 0xFFFF) eco2 = 0;

    /* SDS011 */
    float pm25_now, pm10_now;
    if (readSDS011(pm25_now, pm10_now)) {
      sds_pm25 = pm25_now;
      sds_pm10 = pm10_now;
      sdsOk = true;
    }

    /* UV */
    float uvVoltage = readUVVoltage();
    float uvIndex   = voltageToUVIndex(uvVoltage);

    /* Firebase JSON */
    FirebaseJson json;
    json.set("timestamp_ms", (long long)millis());
    json.set("temperature_c", t);
    json.set("humidity_rh", h);
    json.set("tvoc_ppb", (int)tvoc);
    json.set("eco2_ppm", (int)eco2);
    json.set("mq2", (int)ppm2);
    json.set("mq4", (int)ppm4);
    json.set("mq6", (int)ppm6);
    json.set("mq9", (int)ppm9);

    json.set("pm25_ugm3", sds_pm25);
    json.set("pm10_ugm3", sds_pm10);
    json.set("uv_voltage", uvVoltage);
    json.set("uv_index", uvIndex);

    if (Firebase.ready()) {
      if (Firebase.RTDB.setJSON(&fbdo, FIRE_PATH, &json)) {
        Serial.println(" Firebase Updated");
      } else {
        Serial.print(" Firebase Error: ");
        Serial.println(fbdo.errorReason());
      }
    } else {
      Serial.println(" Firebase not ready");
    }

    Serial.printf(
      "SGP: TVOC=%u ppb eCO2=%u ppm | MQ ppm: %d %d %d %d | SDS011: PM2.5=%.1f PM10=%.1f ug/m3 | UV=%.2f V UVI=%.2f\n",
      tvoc, eco2, (int)ppm2, (int)ppm4, (int)ppm6, (int)ppm9,
      sds_pm25, sds_pm10, uvVoltage, uvIndex
    );
  }
}