from flask import Flask, jsonify, request
import os
from ultralytics import YOLO
import cv2
import torch
from sort import Sort

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

# Cargar modelo yolov8
yolo_roses = YOLO('rosa.pt')

# Inicializar tracker
tracker = Sort()

# Funciones
# Funcion para obtener el numero de clases detectadas
def get_num_classes(res_detection):
    # Obtener nombre de las clases
    clases = res_detection.names
    # Lista para guardar el número de objetos por clase
    num_clases_detection = []
    for k, v in clases.items():
        # Contar los objetos detectados por clase
        num_clases_detection.append(res_detection.boxes.cls.tolist().count(k))
    # Crear un diccionario con las clases y número de objetos detectados
    objetos_detectados = dict(zip(clases.values(), num_clases_detection))
    return objetos_detectados

# Funcion para obtener el path donde se almacenan los resultados
def obtener_path(res):
    # Acceder a la información guardada
    save_dir = res.save_dir
    img_path = res.path
    # Combinar la ruta completa de la imagen con la parte relativa del save_dir
    full_save_path = os.path.join(os.path.dirname(img_path), save_dir)
    return full_save_path

# Funcion para detectar rosas
def detectar(img_path,  model = yolo_roses):
    res = model.predict(img_path, save = True, imgsz=900, conf=0.6, iou=0.7, device = '1')[0]
    return res

#########################################################################################
# Funcion para videos
def detectar_video(video_path, model = yolo_roses):
    cap = cv2.VideoCapture(video_path)
    # Definir clases
    class_names = ['RayaColor', 'PuntoCorte']
    # Definir colores para cada clase
    class_colors = {
        'RayaColor': (95, 76, 17),  # BGR
        'PuntoCorte': (0,0, 139),  # BGR
        }
    # Variables para conteo
    unique_track_ids_rc = set()
    unique_track_ids_pc = set()
    rc_count = 0
    pc_count = 0

    # Obtener información del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crear objeto VideoWriter para guardar el video modificado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "detect_videos/result_video.mp4"
    output_video_name = 'result_video.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height))

    while cap.isOpened():
        # Leer frames de video
        success, frame = cap.read()

        if success:
            line_x = width // 4

            # Dibujar los contadores generales
            cv2.putText(frame, f"RayaColor: {rc_count}", (int(width*0.6), 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f"PuntoCorte: {pc_count}", (int(width*0.6), 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            
            # Dibujar la línea en el cuadro de video
            cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)

            # Inferir con modelo yolo en frames
            results = model(frame, conf = 0.6, stream=True, iou = 0.7, device = '1')
            # Obtener resultados del frame
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                b_conf = result.boxes.conf.cpu().numpy().tolist()
                class_detection = result.boxes.cls.tolist()    
                # Actualizar track        
                tracks = tracker.update(boxes)
                tracks = tracks.astype(int)            
                
                # Graficar resultado de tracks
                for i, (xmin, ymin, xmax, ymax, track_id) in enumerate(tracks):
                    if tracks.any():
                        # Verificar si el objeto cruzó la línea
                        cuadro = xmax - xmin

                        # Calcular el centro horizontal del objeto
                        object_center_x = (xmax + xmin) // 2

                        # Obtener la clase preducha
                        class_index = class_detection[i]
                        class_name = class_names[int(class_index)]
                        # Obtener el color correspondiente a la clase del diccionario
                        color = class_colors.get(class_name, (255, 255, 255))  # Blanco por defecto si la clase no tiene un color asignado
                                    
                        # Dibujar detección
                        cv2.putText(img=frame, text=f"{class_name}-{round(b_conf[i],2)}", org=(xmin+5, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=color, thickness=5)
                        cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=5)
                        
                        # Hacer un conteo el objeto cruce una linea
                        if line_x <= object_center_x <= line_x+cuadro:
                                if class_name == 'RayaColor':
                                    # Agregar id al conjunto de track_ids_rc
                                    unique_track_ids_rc.add(track_id)
                                    # Actualizar el conteo basado en los ids únicos
                                    rc_count = len(unique_track_ids_rc)
                                elif class_name == 'PuntoCorte':
                                    # Agregar id al conjunto de track_ids_pc
                                    unique_track_ids_pc.add(track_id)
                                    # Actualizar el conteo basado en los ids únicos
                                    pc_count = len(unique_track_ids_pc)
                    
            #cv2.imshow("YOLOv8 Inference", frame)
            out.write(frame)
        else:
            return {'RayaColor':rc_count, 'PuntoCorte':pc_count}, output_video_name


# Crear una aplicacion web flask
application = Flask(__name__)

# Ruta de la IA para detectar rosas en imagenes
@application.route("/roses_img/", methods=['GET', 'POST'])

def detection_roses_img():

    # Obtener el texto
    data = request.get_json()
    # Realizar deteccion
    result_detection = detectar(data['img_path'])
    # Contar numero de rosas por estado
    rosas_detectadas = get_num_classes(result_detection)
    # Obtener el path donde se almacena el resultado
    save_path = obtener_path(result_detection)

    return jsonify(
        numero_rosas = rosas_detectadas,
        save_path = save_path.replace('\\', '/')
         )

# Ruta de la IA para detectar rosas en imagenes
@application.route("/roses_vid/", methods=['GET', 'POST'])

def detection_roses_vid():

    # Obtener el texto
    data = request.get_json()
    # Realizar deteccion
    result_detection, output_video_name = detectar_video(data['vid_path'])
    # Obtener el path donde se almacena el resultado
    save_path = r'C:\Users\kvnsg\Documents\Tesis\Deteccion\microservicio\detect_videos'
    output_video_path = os.path.join(save_path, output_video_name)


    return jsonify(
        numero_rosas = result_detection,
        save_path = output_video_path.replace('\\', '/')
         )



if __name__ == "__main__":
    application.run(debug=True)