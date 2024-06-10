import torch
from PIL import ImageDraw, Image
import numpy as np
import pandas as pd
import csv
from torchvision import transforms
from PIL import Image as PILImage

class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

            return resized_image

class TATR:
    def __init__(self, structure_model, device, reader):
        self.structure_model = structure_model
        self.structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device
        self.reader = reader
        self.structure_model.eval()  # Set the model to inference mode

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        width, height = size
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        return boxes

    def outputs_to_objects(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects

    def recognize_table(self, image):
        pixel_values = self.structure_transform(image).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.structure_model(pixel_values)

        # postprocess to get individual elements
        id2label = self.structure_model.config.id2label
        id2label[len(self.structure_model.config.id2label)] = "no object"
        cells = self.outputs_to_objects(outputs, image.size, id2label)

        # visualize cells on cropped table
        draw = ImageDraw.Draw(image)
        for cell in cells:
            draw.rectangle(cell["bbox"], outline="red")

        return image, cells

    def get_cell_coordinates_by_row(self, table_data):
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates


    def apply_ocr(self, cell_coordinates, cropped_table, to_csv=False):
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(cell_coordinates):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))
                # apply OCR
                result = self.reader.readtext(np.array(cell_image))
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                else:
                    text = ""
                row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[str(idx)] = row_text
        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for idx, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[str(idx)] = row_data

        # write to csv
        with open('output.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            for row, row_text in data.items():
                wr.writerow(row_text)
        # return as Pandas dataframe
        df = pd.DataFrame()  # Create an empty DataFrame
        if to_csv:
            try:
                df = pd.read_csv('output.csv')
            except pd.errors.EmptyDataError:
                print("Warning: No data found in the CSV file")
        # df = pd.read_csv('output.csv')
        return df, data

    def process_table_images(self, images):
        # Adjusted implementation of process_pdf to work with table images
        # Multiple tables processing
        results = []
        for cropped_table in images:
            image_processed, cells = self.recognize_table(PILImage.fromarray(cropped_table))
            cell_coordinates = self.get_cell_coordinates_by_row(cells)
            df, data = self.apply_ocr(cell_coordinates, image_processed)
            results.append((image_processed, df, data))
        return results

    def get_tables(self, images):
        table_list = self.process_table_images(images)
        return [pd.DataFrame(tb_tuple[2]) for tb_tuple in table_list]

    def clear_memory(self):
        # Clear memory if needed
        torch.cuda.empty_cache()