import cv2

# Učitavanje modela za lice, osmijeh i oči
lice = cv2.CascadeClassifier('lice.xml')
oko = cv2.CascadeClassifier('oko.xml')
osmijeh = cv2.CascadeClassifier('osmijeh.xml')

# Pokreni web kameru
snimka = cv2.VideoCapture(0)

while True:
    ret, okvir = snimka.read()

    # Pretvorba u crno bijelu snimku
    crno_bijelo = cv2.cvtColor(okvir, cv2.COLOR_BGR2GRAY)

    # Detektiranje lica
    lica = lice.detectMultiScale(crno_bijelo, scaleFactor=1.3, minNeighbors=5)

    # Prolazi kroz svako lice
    for (x, y, w, h) in lica:
        cv2.rectangle(okvir, (x, y), (x+w, y+h), (255, 0, 0), 5)

        roi_crno_bijelo = crno_bijelo[y:y+h, x:x+w]
        roi_boja = okvir[y:y+h, x:x+w]

        # Detektiranje očiju
        oci = oko.detectMultiScale(roi_crno_bijelo, scaleFactor=1.3, minNeighbors=5)

        # Prolazi kroz svako oko
        for (ex, ey, ew, eh) in oci:
            cv2.rectangle(roi_boja, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

        # Detektiranje osmijeha
        osmijesi = osmijeh.detectMultiScale(roi_crno_bijelo, scaleFactor=1.7, minNeighbors=22)

        # Prolazi kroz svaki osmijeh
        for (sx, sy, sw, sh) in osmijesi:
            cv2.rectangle(roi_boja, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 5)

    # Prikaz rezultata
    cv2.imshow('Detektor', okvir)

    # Izađi iz programa prilikom pritiska tipke "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Čišćenje
snimka.release()
cv2.destroyAllWindows()
