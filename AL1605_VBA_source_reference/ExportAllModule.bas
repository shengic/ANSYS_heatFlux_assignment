' run this sub if you want to export all the bas file

Sub ExportAllModules()
    Dim vbProj As VBIDE.VBProject
    Dim vbComp As VBIDE.VBComponent
    Dim exportFolder As String
    
    ' Set the project and export folder
    Set vbProj = ThisWorkbook.VBProject
    exportFolder = "J:\New TPS\front end CAD models\"
    
    ' Create the folder if it doesn't exist
    If Dir(exportFolder, vbDirectory) = "" Then MkDir exportFolder
    
    ' Loop through each component and export it
    For Each vbComp In vbProj.VBComponents
        vbComp.export exportFolder & vbComp.Name & ".bas"
    Next vbComp
End Sub